import pandas as pd
import matplotlib.pyplot as plt
import requests

# get data from fruityvise API
fruityvise_url = "https://www.fruityvice.com/api/fruit/all"
response = requests.get(fruityvise_url)
data = response.json()

#access banana info
banana = requests.get("https://www.fruityvice.com/api/fruit/banana").json()
print(banana)

#access pizza info ERROR
pizza = requests.get("https://www.fruityvice.com/api/fruit/pizza").json()
print(pizza)

# making the data frame and renaming the columns
df = pd.DataFrame(data)
df = df.rename(columns={'name': 'Fruit Name', 'nutritions': 'Nutritional Information'})

def get_sugar(nutrition):
    return nutrition["sugar"]

df["sugar"] = df["Nutritional Information"].apply(get_sugar)

# the .head() method is used to display the 10 rows of the DataFrame with the highest sugar content
top_fruits = df.sort_values("sugar", ascending=False).head(10)

plt.figure(figsize=(15, 8))
plt.bar(top_fruits['Fruit Name'], top_fruits['sugar'], color='orange')
plt.xlabel('Fruit Name')
plt.ylabel('Grams of Sugar')
plt.title('Top 10 Fruits by Sugar Content')
plt.show()
