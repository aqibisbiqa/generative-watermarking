from prettytable import PrettyTable

arr = ["repo | pixel, NO CODING", "bytes_encoded", "accuracy"]

pixel_no_code = PrettyTable()
arr[0] = "repo | pixel, NO CODING"
pixel_no_code.field_names = arr
pixel_no_code.align[arr[0]] = "l"
pixel_no_code.add_rows([
    ["google/ddpm-church-256", "8192", "74.54%"],
    ["google/ddpm-bedroom-256", "8192", "72.54%"],
    ["google/ddpm-cat-256", "8192", "71.62%"],
    ["google/ddpm-celebahq-256", "8192", "75.71%"],
    ["", "", ""],
    ["dboshardy/ddim-butterflies-128", "2048", "99.71%"],
    ["lukasHoel/ddim-model-128-lego-diffuse-1000", "2048", "99.97%"],
])

print(pixel_no_code)

pixel_grs_ham = PrettyTable()
arr[0] = "repo | pixel, GRS + HAM ONLY"
pixel_grs_ham.field_names = arr
pixel_grs_ham.align[arr[0]] = "l"
pixel_grs_ham.add_rows([
    ["google/ddpm-church-256", "2200", "75.53%"],
    ["google/ddpm-bedroom-256", "2200", "71.97%"],
    ["google/ddpm-cat-256", "2200", "70.91%"],
    ["google/ddpm-celebahq-256", "2200", "75.08%"],
    ["", "", ""],
    ["dboshardy/ddim-butterflies-128", "600", "99.93%"],
    ["lukasHoel/ddim-model-128-lego-diffuse-1000", "600", "99.59%"],
])

print(pixel_grs_ham)

pixel_all = PrettyTable()
arr[0] = "repo | pixel, ALL CODES (w/ BRS)"
pixel_all.field_names = arr
pixel_all.align[arr[0]] = "l"
pixel_all.add_rows([
    ["google/ddpm-church-256", "200", "94.86%"],
    ["google/ddpm-bedroom-256", "200", "90.29%"],
    ["google/ddpm-cat-256", "200", "90.33%"],
    ["google/ddpm-celebahq-256", "200", "91.95%"],
])

print(pixel_all)

latent_no_code = PrettyTable()
arr[0] = "repo | latent, NO CODING"
latent_no_code.field_names = arr
latent_no_code.align[arr[0]] = "l"
latent_no_code.add_rows([
    ["runwayml/stable-diffusion-v1-5", "512", "58.16%"],
    ["stabilityai/stable-diffusion-2-1-base", "512", "58.12%"],
    ["friedrichor/stable-diffusion-2-1-realistic", "1152", "63.63%"],
])

print(latent_no_code)

latent_grs_ham = PrettyTable()
arr[0] = "repo | latent, GRS + HAM ONLY"
latent_grs_ham.field_names = arr
latent_grs_ham.align[arr[0]] = "l"
latent_grs_ham.add_rows([
    ["runwayml/stable-diffusion-v1-5", "200", "56.45%"],
    ["stabilityai/stable-diffusion-2-1-base", "200", "56.00%"],
    ["friedrichor/stable-diffusion-2-1-realistic", "400", "61.49%"],
])

print(latent_grs_ham)

latent_all = PrettyTable()
arr[0] = "repo | latent, ALL CODES (w/ BRS)"
latent_all.field_names = arr
latent_all.align[arr[0]] = "l"
latent_all.add_rows([
    ["runwayml/stable-diffusion-v1-5", "200", "49.95%"],
    ["stabilityai/stable-diffusion-2-1-base", "200", "50.09%"],
    ["friedrichor/stable-diffusion-2-1-realistic", "200", "55.51%"],
])

print(latent_all)








# Print the table
