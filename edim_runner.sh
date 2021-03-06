echo Start exclusive representation training
data_base_folder="data"
xp_name="Exclusive_representation_training"
conf_path="conf/exclusive_conf.yaml"
trained_enc_x_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_x/state_dict.pth"
trained_enc_y_path="mlruns/3/38e65dbd8d1246fab33f079e16510019/artifacts/sh_encoder_y/state_dict.pth"

PYTHONPATH=$PYTHONPATH:src python src/edim_train.py --xp_name $xp_name --conf_path $conf_path --data_base_folder $data_base_folder --trained_enc_x_path $trained_enc_x_path --trained_enc_y_path $trained_enc_y_path
