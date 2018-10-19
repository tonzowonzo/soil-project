# Soil data scraper.

# Import required libraries.
import requests
import re
from bs4 import BeautifulSoup
import pandas as pd



dataframe_list = []
start_number = 1500

# Try search through rows.
for i in range(1000):
    # Define the url we're scraping from.
    url = "https://viewer-nsdr.landcareresearch.co.nz/soil/id/nsdr/sa_site/" + str(start_number + i) + "?view=sitereport"
    
    # Get the html info from the page.
    page = requests.get(url)
    
    # Create the html soup from the page.
    soup = BeautifulSoup(page.text, "html.parser")
    
    # Find all of the headings.
    headings = soup.find_all("h1")
    tables = soup.findChildren("table")
    my_table = tables[4]
    rows = my_table.findChildren(["th", "tr"])
    # List to append all of the data to.
    values_list = []
    for row in rows:
        cells = row.findChildren("td")
        for cell in cells:
            value = cell.text
            values_list.append(value)
        print(len(values_list))
        if len(values_list) == 11:
            dataframe_list.append(values_list)
            values_list = []

# Put list into dataframe.
df = pd.DataFrame(dataframe_list)

# Preprocess the data.
# Get failure from list of data.
failure_regex = re.compile(r"failure: \w+")
df["failure"] = [re.findall(failure_regex, i) for i in df[6]]
# Get strength from list of data.
strength_regex = re.compile(r"strength: \w+")
df["strength"] = [re.findall(strength_regex, i) for i in df[6]]
# Get plasticity from list of data.
plasticity_regex = re.compile(r"plasticity: (?:very)|plasticity: (?:non)|plasticity: (?:slightly)|plasticity: (?:moderately)")
df["plasticity"] = [re.findall(plasticity_regex, i) for i in df[6]]

# Turn all of the lists within the dataframe into strings.
df["failure"] = df["failure"].apply(lambda x: "".join(map(str, x)))
df["strength"] = df["strength"].apply(lambda x: "".join(map(str, x)))
df["plasticity"] = df["plasticity"].apply(lambda x: "".join(map(str, x)))
    
# Create an upper and lower bounds feature (horizon top and bottom).
upper_and_lower_limits = [x.split(" - ") for x in df[1]]
df["upper_bound"] = [limit[0] for limit in upper_and_lower_limits]
df["lower_bound"] = [limit[1] for limit in upper_and_lower_limits]

# Get the colours separated (by primary and secondary colour).
colour_regex = re.compile("[a-z]+")
df["colour"] = [re.findall(colour_regex, i) for i in df[2]]
# Separate the primary colour.
df["primary_colour"] = ""
for i, colour in enumerate(df["colour"]):
    if colour:
        df["primary_colour"][i] = colour[-1]
    else:
        df["primary_colour"][i] = ""
# Separate the secondary colour.
df["secondary_colour"] = ""
for i, colour in enumerate(df["colour"]):
    df["secondary_colour"][i] = " ".join(colour[:-1])

        

# Save the dataframe as a csv.
# Rename the columns.
df = df.rename(index=str, columns={0: "designation", 1: "depth", 2: "matrix_colour",
                              3: "mottles", 4: "texture", 5: "structure",
                              6: "cementation", 7: "resistance", 8: "coatings",
                              9: "pan", 10: "boundary"})
# Put plasticity at the end.
columns = ["designation", "depth", "matrix_colour", "mottles", "texture",
           "structure", "cementation", "resistance", "coatings", "pan", 
           "boundary", "failure", "strength", "upper_bound", "lower_bound",
           "colour", "primary_colour", "secondary_colour", "plasticity"]
df = df[columns]
df.to_csv(r"C:/Users/Tim/Desktop/SOILAIPROJECT/soil_plasticity_data.csv")

# Get the dataframe where plasticity isn't empty.
df_plasticity = df[df["plasticity"] != ""]
df_plasticity = df_plasticity[["mottles", "texture",
           "structure", "cementation", "resistance", "coatings", "pan", 
           "boundary", "failure", "strength", "upper_bound", "lower_bound",
           "primary_colour", "secondary_colour", "plasticity"]]
# Encode the data.
char_cols = df_plasticity.dtypes.pipe(lambda x: x[x == "object"]).index
label_mapping = {}
df_encoded = df_plasticity.copy()
for col in char_cols:
    df_encoded[col], label_mapping[col] = pd.factorize(df_encoded[col])
    
df_encoded.to_csv(r"C:/Users/Tim/Desktop/SOILAIPROJECT/soil_plasticity_data_encoded.csv")
