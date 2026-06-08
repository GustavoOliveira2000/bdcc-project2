OUR_BUCKET=gs://bdcc_code_v03

if ! gcloud storage buckets describe ${OUR_BUCKET} >/dev/null 2>&1; then
	gcloud storage buckets create ${OUR_BUCKET} --location=europe-west1
fi


gcloud storage cp bdcc_code/dask/dask.sh ${OUR_BUCKET}/dask/dask.sh
