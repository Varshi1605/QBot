from atlassian import Confluence
import requests
import json
import os

URL="https://quantitativebrokers.atlassian.net"
USER=os.getenv("USER")
pass_= os.getenv("PASS")

confluence = Confluence(url=URL,username=USER,password=pass_)

#getting parent id of the page
parent_id="2828435479"
file_path = "confluence_data.json"

#Load existing data if file exists
if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        confluence_data = json.load(f)
else:
    confluence_data = []

# Track already added page IDs to avoid duplicates
existing_ids = {page["id"] for page in confluence_data}

#Recursive function to fetch and append new content
def fetch_page_and_descendants(page_id):
    if page_id in existing_ids:
        return  # Skip if already added

    page = confluence.get_page_by_id(page_id, expand='body.storage')
    confluence_data.append({
        "id": page['id'],
        "title": page['title'],
        "content": page['body']['storage']['value']
    })
    existing_ids.add(page['id'])

    # Recurse on children
    children = confluence.get_child_pages(page_id)
    for child in children:
        fetch_page_and_descendants(child['id'])

#Start from parent
#fetch_page_and_descendants(parent_id)

# Step 5: Save updated data
with open(file_path, "w", encoding="utf-8") as f:
    json.dump(confluence_data, f, indent=4)

print(f"Updated {file_path} with {len(confluence_data)} total pages.")

#####JIRA API requests

#jql_query="project=QBDEV AND issuetype=Story"
headers={
    "Accept": "application/json"
}

start_at = 0
max_results = 100
all_issues = []

while True:
    params = {
        "jql": jql_query,
        "fields": "summary,description",
        "startAt": start_at,
        "maxResults": max_results
    }

    response = requests.get(
        f"{URL}/rest/api/2/search",
        headers=headers,
        auth=(USER, pass_),
        params=params
    )

    data = response.json()

    issues = data.get("issues", [])
    if not issues:
        break

    for issue in issues:
        all_issues.append({
            "id": issue['key'],
            "title": issue["fields"].get("summary", ""),
            "content": issue["fields"].get("description", "")
        })

    start_at += max_results
    if start_at >= data.get("total", 0):
        break

# Save to file
with open("jira_data.json", "w", encoding="utf-8") as f:
    json.dump(all_issues, f, indent=4)

print(f"Extracted {len(all_issues)} issues. Data extraction done!")
