# candidate_ranking_system
Candidate ranking &amp; recommendation system for modern recruiting platforms

**Instructions:-**

Please use version_3 folder details it is the latest one and also check metadata.yaml file which is present in it. 

If you wanted to generate the top 100 candidates based on JD please follow below steps:

**Step1:-**

Pull docker image using below command

**_Command:_** "docker pull thippeswamyhu/redrob-ranker:v1"

**Step2:-**

create one data folder, then add "candidates.jsonl" and "job_requirements.json"(this file is available in version_3 floder)

**Step3:-**

Before giving run command move back to one folder Example:if you are in this path "E:\IndiaRuns\hackathon\data>" run command in "E:\IndiaRuns\hackathon>" path.

**Step3:-**

**_Execute this command:_** "docker run --rm -v "./data:/data" thippeswamyhu/redrob-ranker:v1"

**Output:-**

Top 100 candidates candidate_recommendations.csv file and candidate_recommendations.xlsx file will be genrated in data folder. (Reason for generating .xlsx file is in the submission portal it is accepting only .xlsx file)



