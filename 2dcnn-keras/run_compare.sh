data="/data/berisha_lab/neuwirth/data/mnf16"
masks="/data/berisha_lab/neuwirth/annotations_3"
checkpoint="/data/berisha_lab/neuwirth/code/2dcnn-keras/checkpoint"
crop=32
classes=5
samples=100000
epochs=1
batch=128

python compare_models.py --data $data --masks $masks/train --test_data $data --test_masks $masks/test --crop $crop --checkpoint $checkpoint --classes $classes --epochs $epochs --batch $batch --balance --samples $samples --validate --valdata $data --valmasks $masks/val --valsamples $samples
