import hashlib,importlib,importlib.util,os,sys
from pathlib import Path

class SolverValidationWorker:
    def install_solver(self, enabled):
        import torch
        parent = os.environ.get("GDN_FLA_PACKAGE", "vllm.model_executor.layers.fla.ops")
        source = Path(os.environ["GDN_SOLVER_SOURCE"])
        name = parent + "._pr_candidate_inverse"
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, source)
            candidate = importlib.util.module_from_spec(spec)
            sys.modules[name] = candidate
            spec.loader.exec_module(candidate)
        candidate = sys.modules[name]
        os.environ["FLA_TRIL_PARALLEL_DIAGONALS"] = "1" if enabled else "0"
        self._pr_solver_tally = {"calls":0,"eligible_calls":0,"enabled":enabled}
        tally = self._pr_solver_tally
        def invoke(*args,**kwargs):
            a=kwargs.get("A", args[0] if args else None)
            tally["calls"] += 1
            if a.shape[0]==1 and a.shape[2:]==(32,64) and a.dtype==torch.float32 and kwargs.get("cu_seqlens") is not None and kwargs.get("output_dtype")==torch.bfloat16 and not candidate.is_tma_supported:
                tally["eligible_calls"] += 1
            return candidate.solve_tril(*args,**kwargs)
        importlib.import_module(parent+".chunk").solve_tril = invoke
        return {"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"enabled":enabled,"torch":str(torch.__version__)}

    def solver_tally(self):
        return self._pr_solver_tally
