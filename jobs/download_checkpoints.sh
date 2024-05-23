mkdir checkpoints
cd checkpoints

pip install gdown

mkdir SQUAD
cd SQUAD
gdown --id 1cQM9dnryw76r9jRXoyPMeDYPpEYqdRZG -O config.json
gdown --id 1C6-l8hl3uv4uJ9tJ-V-mPSSeh4h5l7Lw -O trainer_state.json
gdown --id 1QRJn3LG_yRvwpFXs37P-IXrVLHHPCCW5 -O pytorch_model.bin

cd ..
mkdir IWSLT
cd IWSLT
gdown --id 1spOYi8V8W_wNCzBchJAV00J04u3TA6gn -O config.json
gdown --id 1wlprMZQQLL-BBtlS78y2PX3EQjFadrk5 -O trainer_state.json
gdown --id 1fMP2AzSiwfnO0M3fQGkpeCj3U77cp1bH -O pytorch_model.bin

cd ..
mkdir CNNDM
cd CNNDM
gdown --id 1zsM4g3VC1KXNlCflNtgM3YKcal8KALxT -O config.json
gdown --id 1m11OvlcykbD8TgaIKLNMbiPZk4XCW9-S -O trainer_state.json
gdown --id 1R4n_uiYxQcO3cJ8zlD6XPPXemvcus-bm -O pytorch_model.bin
