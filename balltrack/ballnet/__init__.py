"""ballnet: neural-network ball tracker (CNN encoder + ConvLSTM + heatmap decoder).

Trained on SoccerNet-Tracking (Hugging Face: SoccerNet/SN-Tracking-2023), whose
30-second 1080p clips have a ball box in every frame. Writes the same CSV as
balltrack, so `python -m balltrack evaluate` and `render` work on its output.

    python -m ballnet.prepare E:\\soccernet            # download + convert the training data
    python -m ballnet.train   E:\\soccernet\\prepared_1280x720   # train on the GPU
    python -m ballnet.predict <video> E:\\soccernet\\ballnet.pt
"""
