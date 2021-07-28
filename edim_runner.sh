echo Start exclusive representation training
xp_name="Exclusive_representation_training"
conf_path="conf/exclusive_conf.yaml"
trained_enc_x_path="mlruns/3/463503f630004169a7fcc6bebb836488/artifacts/sh_encoder_x/state_dict.pth"
trained_enc_y_path="mlruns/3/463503f630004169a7fcc6bebb836488/artifacts/sh_encoder_y/state_dict.pth"

PYTHONPATH=$PYTHONPATH:src python src/edim_train.py --xp_name $xp_name --conf_path $conf_path --trained_enc_x_path $trained_enc_x_path --trained_enc_y_path $trained_enc_y_path
