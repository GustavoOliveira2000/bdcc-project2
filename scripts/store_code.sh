SCRIPT_BUCKET=gs://goog-dataproc-initialization-actions-europe-west1
OUR_BUCKET=gs://bdcc_code_v03

# if ! gcloud storage buckets describe ${OUR_BUCKET} >/dev/null 2>&1; then
	gcloud storage buckets create ${OUR_BUCKET} --location=europe-west1
# fi


gcloud storage cp ${SCRIPT_BUCKET}/gpu/install_gpu_driver.sh ${OUR_BUCKET}/gpu/install_gpu_driver.sh
gcloud storage cp ${SCRIPT_BUCKET}/dask/dask.sh ${OUR_BUCKET}/dask/dask.sh
gcloud storage cp ${SCRIPT_BUCKET}/rapids/rapids.sh ${OUR_BUCKET}/rapids/rapids.sh
gcloud storage cp ${SCRIPT_BUCKET}/python/pip-install.sh ${OUR_BUCKET}/python/pip-install.sh
