"""Model-level A/B for the exact upstream solver on the pinned MetaX engine."""
import hashlib,importlib,importlib.util,json,os,sys,time
from pathlib import Path



def main():
    from vllm import LLM,SamplingParams
    started=time.time()
    llm=LLM(model=os.environ["GDN_MODEL_PATH"],tensor_parallel_size=1,dtype="bfloat16",max_model_len=8192,max_num_seqs=2,max_num_batched_tokens=4096,gpu_memory_utilization=0.75,kv_cache_memory_bytes=2147483648,enforce_eager=True,compilation_config=0,enable_prefix_caching=False,language_model_only=True,worker_extension_cls="port_model_worker.SolverValidationWorker")
    questions=[("arithmetic","What is 19 + 23? Reply with only the integer.","42"),("capital","What is the capital of France? Reply with only the city name.","Paris"),("multiplication","What is 7 times 8? Reply with only the integer.","56"),("sequence","Complete this sequence: 2, 4, 6, 8, __. Reply with only the integer.","10"),("json","Return exactly a JSON object with the key answer and integer value 42.",None),("chinese","中国的首都是哪里？只回答城市名。","北京")]
    for copies in (128,512):
        questions.append(("long_"+str(copies),"Reference notes:\n"+"The archive records ordinary events in chronological order.\n"*copies+"\nIgnore the notes. What is 19 + 23? Reply with only the integer.","42"))
    messages=[[{"role":"user","content":q[1]}] for q in questions]
    params=SamplingParams(temperature=0,max_tokens=128,seed=20261008)
    result={"scope":"Exact upstream solver overlay on pinned MetaX e434d79 engine, not a complete current-upstream runtime build or official benchmark accuracy", "questions":[],"variants":{}}
    runs={}
    for enabled in (False,True):
        key="candidate" if enabled else "baseline"
        identity=llm.collective_rpc("install_solver",args=(enabled,),timeout=60)
        outputs=llm.chat(messages,sampling_params=params,chat_template_kwargs={"enable_thinking":False},use_tqdm=False)
        runs[key]=outputs
        result["variants"][key]={"identity":identity,"execution":llm.collective_rpc("solver_tally",timeout=60)}
        print("VARIANT_COMPLETE",json.dumps(result["variants"][key]),flush=True)
    for i,(name,prompt,expected) in enumerate(questions):
        b,c=runs["baseline"][i],runs["candidate"][i]
        bo,co=b.outputs[0],c.outputs[0]
        result["questions"].append({"name":name,"prompt_tokens":len(b.prompt_token_ids),"baseline_text":bo.text,"candidate_text":co.text,"baseline_tokens":list(bo.token_ids),"candidate_tokens":list(co.token_ids),"exact_token_match":list(bo.token_ids)==list(co.token_ids),"finish_reasons":[bo.finish_reason,co.finish_reason],"expected":expected,"baseline_answer_matches":expected is None or expected in bo.text,"candidate_answer_matches":expected is None or expected in co.text})
    result["all_token_sequences_match"]=all(q["exact_token_match"] for q in result["questions"])
    result["candidate_exercised"]=all(x["eligible_calls"]>0 and x["enabled"] for x in result["variants"]["candidate"]["execution"])
    result["elapsed_s"]=time.time()-started
    Path(os.environ.get("GDN_RESULTS_PATH", "model-results.json")).write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    assert result["candidate_exercised"] and result["all_token_sequences_match"],result
    print("MODEL_OVERLAY_PASS",flush=True)
    llm.llm_engine.engine_core.shutdown()
    os._exit(0)

if __name__=="__main__":main()
