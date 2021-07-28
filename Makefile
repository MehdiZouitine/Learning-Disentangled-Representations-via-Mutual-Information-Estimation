share_train:
	bash sdim_runner.sh

exclusive_train:
	bash edim_runner.sh

board:
	pkill gunicorn || true
	mlflow ui
	