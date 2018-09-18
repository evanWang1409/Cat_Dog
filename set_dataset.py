import os, re, shutil, argparse

def add_args():
	parser = argparse.ArgumentParser(description='Create a torch dataset for image classification')
	parser.add_argument('--data_dir', dest='data_dir',
						default = '/Users/evnw/Research/Cats_v_Dogs/data')
	args = parser.parse_args()

	return args

def set_dataset(args):
	train_dir = os.path.join(args.data_dir, 'train')
	test_dir = os.path.join(args.data_dir, 'test')
	new_train_dir = os.path.join(args.data_dir, 'train_by_class')
	new_test_dir = os.path.join(args.data_dir, 'test_by_class')

	for new_dir in new_test_dir, new_train_dir:
		if not os.path.exists(new_dir):
			os.mkdir(new_dir)

	put(train_dir, new_train_dir)
	#put(test_dir, new_test_dir)

def put(ori_dir, new_dir):
	cat_dir = os.path.join(new_dir, 'cat')
	dog_dir = os.path.join(new_dir, 'dog')

	for new_dir in cat_dir, dog_dir:
		if not os.path.exists(new_dir):
			os.mkdir(new_dir)

	for img in os.listdir(ori_dir):
		if 'cat' in img:
			new_img = re.sub(r"cat\.", 'cat_', img)
			img_path = os.path.join(ori_dir, img)
			new_img_path = os.path.join(cat_dir, new_img)
			shutil.copy2(img_path, new_img_path)
			print(new_img_path)
		else:
			new_img = re.sub(r"dog\.", 'dog_', img)
			img_path = os.path.join(ori_dir, img)
			new_img_path = os.path.join(dog_dir, new_img)
			shutil.copy2(img_path, new_img_path)
			print(new_img_path)

if __name__ == '__main__':
	args = add_args()
	set_dataset(args)


