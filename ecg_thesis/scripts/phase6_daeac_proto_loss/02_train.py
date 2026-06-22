from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parents[1] / "phase6_daeac_proto_bank" / "02_train.py"), run_name="__main__")
