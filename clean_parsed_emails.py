# necessary imports
import pandas as pd
# import the parsed_emails.csv file
df = pd.read_csv("parsed_emails.csv")

# keep only the xOrigin and content columns
# this reduces the size of the dataframe
df = df[['xOrigin', 'content','xFrom']]

# trim out any null xOrigin rows and reset index
df = df.dropna(subset=['xOrigin']).reset_index(drop=True)
# drop rows with empty contents and reset index
df = df.dropna(subset=['content']).reset_index(drop=True)


# save the cleaned dataframe to a csv file
df.to_csv("parsed_emails_cleaned.csv", index=False)