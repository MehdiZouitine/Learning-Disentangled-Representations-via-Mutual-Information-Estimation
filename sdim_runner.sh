echo Start shared representation training
xp_name="Share_representation_training"
conf_path="conf/share_conf.yaml"

PYTHONPATH=$PYTHONPATH:src python src/sdim_train.py --xp_name $xp_name --conf_path $conf_path 
