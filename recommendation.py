# The data used in this project is from https://gist.github.com/jeremystan/c3b39d947d9b88b3ccff3147dbcf6c6b
# or a kaggle competition at https://www.kaggle.com/competitions/instacart-market-basket-analysis/data?select=aisles.csv.zip

# 1. Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict

# from google.colab import drive
# drive.mount('/content/drive')
# cd /content/drive/MyDrive/xxxxxxxx # change xxxxx here with your directory

# dimension tables
aisles = pd.read_csv('aisles.csv')
departments = pd.read_csv('departments.csv')
orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')
# fact tables
order_products_prior = pd.read_csv('order_products_prior.csv')
order_products_train = pd.read_csv('order_products_train.csv')

# print(aisles.shape)
# print(departments.shape)
# print(order_products_prior.shape)
# print(order_products_train.shape)
# print(orders.shape)
# print(products.shape)

aisles.head()
# aisle_id: aisle identifier
# aisle: the name of the aisle

departments.head()
# department_id: department identifier
# department: the name of the department

orders.head()
# order_id: order identifier
# user_id: customer identifier
# eval_set: which evaluation set this order belongs in (see SET described below)
# order_number: the order sequence number for this user (1 = first, n = nth)
# order_dow: the day of the week the order was placed on
# order_hour_of_day: the hour of the day the order was placed on
# days_since_prior: days since the last order, capped at 30 (with NANs for order_number = 1)

products.head()
# product_id: product identifier
# product_name: name of the product
# aisle_id: foreign key
# department_id: foreign key

order_products_prior.head()
# reordered: 1 if this product has been ordered by this user in the past, 0 otherwise
# "prior": orders prior to that users most recent order

order_products_train.head()
# "train": training data supplied to participants

# 2. Data Exploration

# Build prior order details table for data exploration
prior_order_details = order_products_prior.merge(orders, on="order_id")
# prior_order_details.head()

# Covert dow to string
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Sunday" if x==0 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Monday" if x==1 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Tuesday" if x==2 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Wednesday" if x==3 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Thursday" if x==4 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Friday" if x==5 else x)
prior_order_details["order_dow"] = prior_order_details["order_dow"].apply(lambda x:"Saturday" if x==6 else x)
# prior_order_details.head()

# Frequency of Order Based on Days
ax = sns.countplot(x="order_dow",data=prior_order_details,
                   order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
plt.title("Order Frequency (Days)")
plt.xlabel("")
plt.ylabel("Number of Order")
plt.show()

# Visualize order frequency on HoD
order_hours_counts = orders.groupby("order_id")["order_hour_of_day"].mean().reset_index()
order_hod_stats = order_hours_counts.order_hour_of_day.value_counts()

order_hod_stats = order_hod_stats.reset_index()
order_hod_stats.columns = ['Order Time (hours)', 'Number of Order']

sns.barplot(x='Order Time (hours)', y='Number of Order', data=order_hod_stats)
plt.title("Order Hour")
plt.ylabel("Number of Order")
plt.xlabel('Order Time (hours)')
plt.show()

# Reorder Pattern
reorder_heatmap = prior_order_details.groupby(["order_dow", "order_hour_of_day"])["reordered"].mean().reset_index()
reorder_heatmap = reorder_heatmap.pivot('order_dow', 'order_hour_of_day', 'reordered')
plt.figure(figsize=(12,6))
sns.heatmap(reorder_heatmap,cmap="Reds")
plt.title("Reorder Ratio")
plt.ylabel("")
plt.xlabel("Hours")
plt.show()

# the most popular products, aisles and departments
prior_order_all = order_products_prior.merge(products, on="product_id")
prior_order_all.head()

# top 10 products
prior_order_all['product_name'].value_counts().head(10)

# top 10 aisles
prior_order_all[["aisle_id"]].merge(aisles, on="aisle_id")['aisle'].value_counts().head(10)

# top 10 departments
prior_order_all[["department_id"]].merge(departments, on="department_id")['department'].value_counts().head(10)


# 3. Data Quality Check

# Validate the `days_since_prior_order` column in orders table
print("Size of the order dataset: ", orders.shape[0])
print("NaN count in days_since_prior_order column: ", orders[orders.days_since_prior_order.isnull()].shape[0])
print("order_number 1 count in orders table: ", orders[orders.order_number == 1].drop_duplicates().shape[0])
print("user_id count in orders table: ", orders.user_id.drop_duplicates().shape[0])
# number of orders with no prior matches the days since

# Validate all prior orders match in the prior table

orders.groupby(['eval_set'], as_index=False).agg(OrderedDict([('order_id','nunique')]))
print("order_id count in prior: ", order_products_prior['order_id'].nunique())
print("order_id from prior found in orders: ", order_products_prior[order_products_prior.order_id.isin(orders.order_id)].order_id.nunique())

# Validate all train orders match in the train table
print("orders count in train: ", order_products_train['order_id'].nunique())
print("order_id from train found in orders: ", order_products_train[order_products_train.order_id.isin(orders.order_id)].order_id.nunique())

# check intersection between prior and train
print("order_id intersection between prior and train: ", pd.merge(order_products_prior, order_products_train, on = ['order_id']).shape[0])

# users in contained both datasets
orders.groupby(['eval_set'], as_index=False).agg(OrderedDict([('user_id','nunique')]))

prior_user_ids = set(orders[orders['eval_set'] == 'prior']['user_id'])
train_user_ids = set(orders[orders['eval_set'] == 'train']['user_id'])
print("user_ids in prior: ", len(prior_user_ids))
print("user_ids in train: ", len(train_user_ids))
print("intersection of prior and train: ", len(prior_user_ids.intersection(train_user_ids)))

# order counts from each user in the train dataset
(orders[orders.user_id.isin(train_user_ids)][orders.eval_set == 'train']
  .groupby(['user_id'], as_index=False)
  .agg(OrderedDict([('order_number','count')]))
  .rename(columns={'order_number':'order_counts'})).sort_values(by=['order_counts']).head()

# Validate the relative order of `order_num` in prior and train dataset
df_prior_order_max = (orders[orders.user_id.isin(prior_user_ids)][orders.eval_set == 'prior']
  .groupby(['user_id'], as_index=False)
  .agg(OrderedDict([('order_number','max')]))
  .rename(columns={'order_number':'prior_order_max'}))

df_train_order_min = (orders[orders.user_id.isin(train_user_ids)][orders.eval_set == 'train']
  .groupby(['user_id'], as_index=False)
  .agg(OrderedDict([('order_number','min')]))
  .rename(columns={'order_number':'train_order_min'}))

# make sure all prior orders came before the training orders
df_order_diff = pd.merge(df_prior_order_max, df_train_order_min, on = ['user_id'])
print("Rows count where prior_order_max >= train_order_min: ",
      df_order_diff[df_order_diff.prior_order_max >= df_order_diff.train_order_min].shape[0])

# 4 Construct Model Label

# We want user who has reordered in the past (in the prior set), and reordered recently (in the training set), 
# to ensure the interest of purchasing the item, and use it as the label when evaluate the model.
# 
# When constructing the model, only the features from the prior set will be used to 
# train the model avoid data leakage that would give inaccurate data performance.
#

#  all train data orders, set each order on a product from a user as the "unique key"
train_details = order_products_train.merge(orders, on = 'order_id')
train_unique_key = train_details['user_id'].astype('str') + '_' + train_details['product_id'].astype('str')
train_user_ids = set(orders[orders['eval_set'] == 'train']['user_id'])
prior_details = order_products_prior.merge(orders, on = 'order_id')

# do the same thing with prior data.
model_all_data = prior_details[prior_details.user_id.isin(train_user_ids)][['user_id','product_id','order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order']]
model_all_data = model_all_data.drop_duplicates(subset=['user_id', 'product_id'])
model_all_data['unique_key'] = model_all_data['user_id'].astype('str') + '_' + model_all_data['product_id'].astype('str')

model_all_data['label'] = 0

# set the label if the user bought in both train and prior using unique key
model_all_data.loc[model_all_data.unique_key.isin(train_unique_key), 'label'] = 1

model_all_data.head()


