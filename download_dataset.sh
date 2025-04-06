folder_name="amazon_dataset"

mkdir $folder_name
curl -L -o $folder_name/amazon-fine-food-reviews.zip https://www.kaggle.com/api/v1/datasets/download/snap/amazon-fine-food-reviews
unzip $folder_name/amazon-fine-food-reviews.zip -d $folder_name
rm $folder_name/amazon-fine-food-reviews.zip
