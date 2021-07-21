import json

fit = open('/data/polyvore/train_no_dup.json', 'r')
# fit = open('/data/polyvore/test_no_dup.json', 'r')
# fit = open('/data/polyvore/valid_no_dup.json', 'r')
mmm = json.load(fit)

fittt = open('/data/category_summarize_100.json', 'r')
cat_100 = json.load(fittt)
category = []
for cat in cat_100:
	category.append(cat['id'])

train_new_hy = []
for outfits in mmm:
	outfit = outfits['items']
	t = []
	cattt = []
	for m in outfit:
		if m['categoryid'] in category and m['categoryid'] not in cattt:
			cattt.append(m['categoryid'])
			t.append(m)
	outfits['items'] = t
	train_new_hy.append(outfits)
	
train_list = []
for outfit in train_new_hy:
	length = len(outfit['items'])
	if length >= 3:
		train_list.append(outfit)

# with open('/data/train_no_dup_bilstm.json', 'w') as f1:
#     json.dump(train_list, f1)