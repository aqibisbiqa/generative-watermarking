from prettytable import PrettyTable

# Create a PrettyTable instance
table = PrettyTable()

# Define columns
table.field_names = ["pixel/latent", "repo", "bytes_encoded", "accuracy"]

# Add data rows
table.add_row(["pixel", "google/ddpm-church-256", "2200", "75.53%"])
table.add_row(["pixel", "google/ddpm-bedroom-256", "2200", "71.97%"])
table.add_row(["pixel", "google/ddpm-cat-256", "2200", "70.91%"])
table.add_row(["pixel", "google/ddpm-celebahq-256", "2200", "75.08%"])
table.add_row(["pixel", "dboshardy/ddim-butterflies-128", "600", "99.93%"])
table.add_row(["pixel", "lukasHoel/ddim-model-128-lego-diffuse-1000", "600", "99.59%"])
table.add_row(["latent", "runwayml/stable-diffusion-v1-5", "200", "56.45%"])
table.add_row(["latent", "stabilityai/stable-diffusion-2-1-base", "200", "56.00%"])
table.add_row(["latent", "friedrichor/stable-diffusion-2-1-realistic", "400", "61.49%"])

# Print the table
print(table)
