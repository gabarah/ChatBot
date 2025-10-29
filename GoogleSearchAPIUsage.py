from serpapi import GoogleSearch

# Ask user for input
query = input("Enter your search query: ")

# Initialize parameters
params = {
    "engine": "google",           # search engine (google, bing, youtube, etc.)
    "q": query,                   # query entered by user
    "api_key": "28b6d15e47f8f82b6ac4c2a4a7027d89efeac940a3cb11685b0ed3afdcdafc71" # replace with your actual key
}

# Perform the search
search = GoogleSearch(params)
results = search.get_dict()

# Print some useful parts of the result
print("\nTop results:\n")
if "organic_results" in results:
    for idx, result in enumerate(results["organic_results"][:5], start=1):
        title = result.get("title", "No title")
        link = result.get("link", "No link")
        snippet = result.get("snippet", "No snippet")
        print(f"{idx}. {title}\n   {link}\n   {snippet}\n")
else:
    print("No results found or API limit exceeded.")