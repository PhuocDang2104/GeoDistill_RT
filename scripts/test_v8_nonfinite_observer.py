"""Isolated CPU fixture: bad objective aborts before optimizer/checkpoint mutation."""
from pathlib import Path
from unittest.mock import patch
import json
import sys
import tempfile
import torch


def main():
    root=Path(__file__).resolve().parents[1]
    code=root/"drive_upload/AnchorFlow_v8_Dynamics"
    sys.path.insert(0,str(code))
    import run
    import diagnose_anchorflow_v8_nonfinite as debug
    from data import write_json
    torch.set_num_threads(2)
    rgb,sparse,mask,K=run.sample_inputs(torch.device("cpu"),64,128)
    batch={"rgb":rgb,"sparse":sparse,"mask":mask,"K":K,"sid":["synthetic-test"],
           "gt":torch.full_like(mask,25.),"gt_mask":(torch.rand_like(mask)<.2).float(),
           "teacher":torch.full_like(mask,22.),"confidence":torch.full_like(mask,.8)}
    class Loader(list): dataset=[0]
    cfg=json.loads((code/"config.json").read_text())
    objective=run.objective
    def injected_bad(*args,**kwargs):
        loss,stats=objective(*args,**kwargs)
        loss=loss*loss.new_tensor(float("nan"))
        stats["total"]=loss.detach()
        return loss,stats
    with tempfile.TemporaryDirectory(prefix="v8_nonfinite_fixture_") as temp:
        base=Path(temp)
        cfg.update(work=str(base/"work"),drive_runs=str(base/"drive"),epochs=1,encoder_pretrained=False,
                   fused_adamw=False,amp="fp32",workers=0,log_every=1)
        write_json(base/"work/data_contract.json",{"fixture":True})
        config=base/"fixture.json"; write_json(config,cfg)
        with patch.object(sys,"argv",["debug","--code",str(code),"--config",str(config)]), \
             patch.object(run,"device_setup",return_value=torch.device("cpu")), \
             patch.object(run,"data_loader",return_value=Loader([batch])), \
             patch.object(torch.cuda,"get_device_name",return_value="CPU fixture"), \
             patch.object(run,"objective",side_effect=injected_bad), \
             patch.object(torch.optim.AdamW,"step") as optimizer_step:
            try: debug.main()
            except RuntimeError as error: assert "First nonfinite objective" in str(error),str(error)
            else: raise AssertionError("Expected nonfinite abort")
            optimizer_step.assert_not_called()
        folder=base/"drive"/cfg["run_name"]/"metric_kd"
        report=json.loads(next((folder/"nonfinite_debug").glob("*/nonfinite_report.json")).read_text())
        assert report["bad_loss_or_stat_names"]==["total"]
        assert report["batch"]==1
        assert not report["nonfinite_preforward_state_keys"]
        assert all("error" not in p for p in report["same_batch_precision_probes"].values()),report
        assert not (folder/"last.pth").exists()
        print("PASS: first bad objective captured before optimizer; pre-forward state reload valid; report saved; no last checkpoint overwritten.")


if __name__=="__main__": main()
