python3 yolov5/val.py --weights /home/muchun/yolov5/lightning_logs/version_2/detseg_last.pt \
                --data data/electronNetProject.yaml \
                --project ele/val \
                --save-txt \
                --save-conf \
                --device 0
