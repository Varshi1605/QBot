from bs4 import BeautifulSoup
import json

# Clean HTML tags
def clean_html(content):
    return BeautifulSoup(content, "html.parser").get_text()

# JSON data 
with open("confluence_data.json","r") as f1,open("jira_data.json","r") as f2:
    data=json.load(f1)
    jdata=json.load(f2)

# Process conf data
for entry in data:
    if entry["content"]==None:
        entry["content"]=entry["title"]
    entry["content"] = clean_html(entry["content"])

#Process Jira data
for entry in jdata:
    if entry["content"]==None:
        entry["content"]=entry["title"]
    entry["content"]=clean_html(entry["content"])


#save it back now
with open("cleaned_conf_data.json","w",encoding="utf-8") as f1, open("cleaned_jira_data.json","w",encoding="utf-8") as f2:
    json.dump(data,f1,indent=4)
    json.dump(jdata,f2,indent=4)

print("Data preprocessing done!")
