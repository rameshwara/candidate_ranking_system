from time import time

import_time = time()

from os import cpu_count
from json import loads, load

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from tqdm import tqdm
from numpy import array, unique, max as np_max, mean, exp, log, argmax
from numpy.linalg import norm
from pandas import DataFrame

print(f"\nModules import time: {(time() - import_time):.2f} secs.\n")

class CandidateRankingSystem:

    def __init__(self, candidates_data_file, job_requirements_file, embedding_model_folder, llm_model_file):
        
        start = time()
        
        self.candidates_data_file = candidates_data_file
        self.job_requirements_file = job_requirements_file
        
        self.embedding_model = SentenceTransformer(embedding_model_folder, device="cpu")
        self.llm_model = Llama(model_path=llm_model_file, n_ctx=512, n_threads=cpu_count(), verbose=False)
        
        print(f"\nInitial Setup Time: {(time() - start):.2f} secs.\n")
        
    def load_candidates_data(self):
        
        start = time()
        
        self.candidates_data = []

        with open(self.candidates_data_file, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                self.candidates_data.append(loads(line))
                
        print(f"\nLoaded Candidates Data in {(time() - start):.2f} secs.\n")
        
    def load_job_requirements(self):
        
        start = time()
        
        with open(self.job_requirements_file, "r", encoding="utf-8") as f:
            jd_requirements = load(f)
                
        self.ideal_exp_min, self.ideal_exp_max = [int(exp) for exp in jd_requirements["ideal_experience_duration"].split(",")]
        self.exp_min, self.exp_max = [int(exp) for exp in jd_requirements["experience_duration"].split(",")]

        self.desired_experience_areas = [exp_area.strip() for exp_area in jd_requirements["primary_experience_areas"].split(";")]
        self.nice_experience_areas = [exp_area.strip() for exp_area in jd_requirements["secondary_experience_areas"].split(";")]
        self.undesired_experience_areas = [exp_area.strip() for exp_area in jd_requirements["not_preferred_experience"].split(";")]

        self.education_fields = [exp_area.strip() for exp_area in jd_requirements["education_field"].split(";")]
        
        self.desired_skills = jd_requirements["preferred_skills"]
        self.undesired_skills = [exp_area.strip() for exp_area in jd_requirements["not_preferred_skills"].split(";")]

        self.preferred_languages = [lang.strip() for lang in jd_requirements["language"].split(",")]
        
        self.primary_work_location = jd_requirements["primary_work_location"].split(",")
        self.secondary_work_location = jd_requirements["secondary_work_location"].split(",")

        self.preferred_job_titles = jd_requirements["current_job_title"].split(",")
        self.undesired_job_roles = jd_requirements["undesired_roles"].split(",")
        self.preferred_work_industries = jd_requirements["industry"].split(",")

        self.international_candidates_consideration = jd_requirements["international_candidates_consideration"]
        self.international_candidates_relocation_support = jd_requirements["international_relocation_support"]

        self.notice_period_min_days, self.notice_period_max_days = [int(days) for days in jd_requirements["notice_period"].split(",")]
        
        desired_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.desired_experience_areas])
        desired_education = "\n".join([f"* {education.capitalize()}" for education in self.education_fields])
        desired_skills = "\n".join([f"* {skill['skill_id']} ({skill['proficiency']})" for skill in self.desired_skills["core_required"]])
        desired_job_roles = "\n".join([f"* {role}" for role in self.preferred_job_titles])
        nice_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.nice_experience_areas])
        undesired_experiences = "\n".join([f"* {experience.capitalize()}" for experience in self.undesired_experience_areas])
        undesired_skills = "\n".join([f"* {skill.capitalize()}" for skill in self.undesired_skills])
        undesired_job_roles = "\n".join([f"* {role}" for role in self.undesired_job_roles])
        
        self.job_description = f"""
Preferred experience areas:
{desired_experiences}
Preferred Education:
{desired_education}
Preferred Skills:
{desired_skills}
Preferred Job Roles (Past or current):
{desired_job_roles}
Nice to have experience areas:
{nice_experiences}
        """
        
        print(f"Loaded Job Requirements in {(time() - start):.2f} secs.\n")
    
    def generate_embeddings(self, texts, batch_size=8, normalize=True, verbose=True):
        
        start = time()

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=verbose
        )

        embeddings = array([vector.tolist() for vector in embeddings])
        
        print(f"Generated {len(texts)} embedding vectors in {(time() - start):.2f} secs.\n")

        return embeddings
    
    def cosine_similarity(self, query_embs, context_embs):
    
        dot_products = query_embs @ context_embs.T
        query_embs_norms = norm(query_embs, axis=1, keepdims=True)
        context_embs_norms = norm(context_embs, axis=1, keepdims=True)

        cosine_sims = dot_products / (query_embs_norms @ context_embs_norms.T)

        return cosine_sims
    
    def prepare_embeddings(self, potential_candidates):
        
        start = time()
        
        self.candidate_education_fields, self.candidate_skills, self.candidate_certifications, self.candidate_languages = [], [], [], []
        self.candidate_titles, self.candidate_industries, self.candidate_profile_summaries, self.candidate_career_descs = [], [], [], []

        for candidate in tqdm(potential_candidates, desc="Filtering unique entites: "):

            for profile in candidate["profile"]:
                self.candidate_profile_summaries.append(candidate["profile"]["summary"])

            for career in candidate["career_history"]:
                self.candidate_industries.append(career["industry"])
                self.candidate_titles.append(career["title"])
                self.candidate_career_descs.append(career["description"])

            for education in candidate["education"]:
                self.candidate_education_fields.append(f"{education['degree']}, {education['field_of_study']}")

            if candidate["skills"]:
                for skill in candidate["skills"]:
                    self.candidate_skills.append(skill)

            if candidate["certifications"]:
                for certification in candidate["certifications"]:
                    self.candidate_certifications.append(certification)

            if candidate["languages"]:
                for language in candidate["languages"]:
                    self.candidate_languages.append(language)
                    
        print("\nBuilding embeddings on candidate profile summaries:")
        self.candidate_profile_summaries = sorted(unique(self.candidate_profile_summaries).tolist())
        self.candidate_profile_summaries_embs = self.generate_embeddings(texts=self.candidate_profile_summaries)
        self.candidate_profile_summaries_embs = {summary: self.candidate_profile_summaries_embs[si] for si, summary in enumerate(self.candidate_profile_summaries)}

        print("Building embeddings on candidate career descriptions:")
        self.candidate_career_descs = sorted(unique(self.candidate_career_descs).tolist())
        self.candidate_career_descs_embs = self.generate_embeddings(texts=self.candidate_career_descs)
        self.candidate_career_descs_embs = {description: self.candidate_career_descs_embs[di] for di, description in enumerate(self.candidate_career_descs)}

        print("Building embeddings on desired experience areas:")
        self.desired_experience_embs = self.generate_embeddings(texts=self.desired_experience_areas)

        print("Building embeddings on nice to have experience areas:")
        self.nice_experience_embs = self.generate_embeddings(texts=self.nice_experience_areas)

        print("Building embeddings on undesired experience areas:")
        self.undesired_experience_embs = self.generate_embeddings(texts=self.undesired_experience_areas)

        print("Building embeddings on preferred industries of work:")
        self.preferred_industry_embs = self.generate_embeddings(texts=self.preferred_work_industries)

        print("Building embeddings on available candidate industries of work:")
        self.candidate_industries = sorted(unique(self.candidate_industries).tolist())
        self.candidate_industries_embs = self.generate_embeddings(texts=self.candidate_industries)
        self.candidate_industries_embs = {industry: self.candidate_industries_embs[ii] for ii, industry in enumerate(self.candidate_industries)}

        print("Building embeddings on preferred job titles:")
        self.preferred_title_embs = self.generate_embeddings(texts=self.preferred_job_titles)

        print("Building embeddings on undesired job titles:")
        self.undesired_title_embs = self.generate_embeddings(texts=self.undesired_job_roles)

        print("Building embeddings on available candidate titles:")
        self.candidate_titles = sorted(unique(self.candidate_titles).tolist())
        self.candidate_title_embs = self.generate_embeddings(texts=self.candidate_titles)
        self.candidate_title_embs = {title: self.candidate_title_embs[ti] for ti, title in enumerate(self.candidate_titles)}

        print("Building embeddings on preferred education degrees:")
        self.preferred_education_embs = self.generate_embeddings(texts=self.education_fields)

        print("Building embeddings on available candidate education degrees:")
        self.candidate_education_fields = sorted(unique(self.candidate_education_fields).tolist())
        self.candidate_education_embs = self.generate_embeddings(texts=self.candidate_education_fields)
        self.candidate_education_embs = {edu_field: self.candidate_education_embs[ei] for ei, edu_field in enumerate(self.candidate_education_fields)}

        print("Building embeddings on preferred primary / required skills:")
        self.preferred_primary_skill_names = sorted(unique([skill["skill_id"] for skill in self.desired_skills["core_required"]]).tolist())
        self.preferred_primary_skill_embs = self.generate_embeddings(texts=self.preferred_primary_skill_names)
        self.preferred_primary_skill_embs = {skill: self.preferred_primary_skill_embs[si] for si, skill in enumerate(self.preferred_primary_skill_names)}
        self.primary_skills_meta = {skill["skill_id"]: skill for skill in self.desired_skills["core_required"]}

        print("Building embeddings on preferred secondary / optional skills:")
        self.preferred_secondary_skill_names = sorted(unique([skill["skill_id"] for skill in self.desired_skills["secondary_skills"]]).tolist())
        self.preferred_secondary_skill_embs = self.generate_embeddings(texts=self.preferred_secondary_skill_names)
        self.preferred_secondary_skill_embs = {skill: self.preferred_secondary_skill_embs[si] for si, skill in enumerate(self.preferred_secondary_skill_names)}
        self.secondary_skills_meta = {skill["skill_id"]: skill for skill in self.desired_skills["secondary_skills"]}

        print("Building embeddings on available candidate skills:")
        self.candidate_skills = sorted(unique([skill["name"] for skill in self.candidate_skills]).tolist())
        self.candidate_skill_embs = self.generate_embeddings(texts=self.candidate_skills)
        self.candidate_skill_embs = {skill: self.candidate_skill_embs[si] for si, skill in enumerate(self.candidate_skills)}

        print("Building embeddings on available candidate certifications:")
        self.candidate_certifications = sorted(unique([cert["name"] for cert in self.candidate_certifications]).tolist())
        self.candidate_cert_embs = self.generate_embeddings(texts=self.candidate_certifications)
        self.candidate_cert_embs = {cert: self.candidate_cert_embs[ci] for ci, cert in enumerate(self.candidate_certifications)}
        
        print(f"Total embeddings preparation time: {(time() - start):.2f} secs.\n")
        
    def profile_score(self, profile):
    
        summary = profile["summary"]
        current_title, current_industry = profile["current_title"], profile["current_industry"]
        experience_years = profile["years_of_experience"]

        summary_emb = self.candidate_profile_summaries_embs[summary].reshape(1, -1)
        current_title_emb = self.candidate_title_embs[current_title].reshape(1, -1)
        current_industry_emb = self.candidate_industries_embs[current_industry].reshape(1, -1)

        experience_years_score = (experience_years - self.exp_min) / (self.exp_max - self.exp_min)
        current_title_score = np_max(self.cosine_similarity(current_title_emb, self.preferred_title_embs)[0])
        current_industry_score = np_max(self.cosine_similarity(current_industry_emb, self.preferred_industry_embs)[0])
        current_title_irrelevance = 1 - np_max(self.cosine_similarity(current_title_emb, self.undesired_title_embs)[0])

        desired_exp_score = np_max(self.cosine_similarity(summary_emb, self.desired_experience_embs)[0])
        nice_exp_score = np_max(self.cosine_similarity(summary_emb, self.nice_experience_embs)[0])
        undesired_exp_score = 1 - np_max(self.cosine_similarity(summary_emb, self.undesired_experience_embs)[0])

        return mean([experience_years_score, current_title_score, current_industry_score, current_title_irrelevance, desired_exp_score, nice_exp_score, undesired_exp_score])

    def career_score(self, career):

        scores = []
        relevant_experience_yrs = 0
        current_exp = []
        relevant_careers = []

        for role in career:

            role_description, role_industry, is_current = role["description"], role["industry"], role["is_current"]
            role_title, role_duration = role["title"], round(role["duration_months"] / 12, ndigits=1)

            description_emb = self.candidate_career_descs_embs[role_description].reshape(1, -1)
            desired_exp_score = np_max(self.cosine_similarity(description_emb, self.desired_experience_embs)[0])
            nice_exp_score = np_max(self.cosine_similarity(description_emb, self.nice_experience_embs)[0])
            undesired_exp_score = 1 - np_max(self.cosine_similarity(description_emb, self.undesired_experience_embs)[0])

            current_title_emb = self.candidate_title_embs[role_title].reshape(1, -1)
            current_title_score = np_max(self.cosine_similarity(current_title_emb, self.preferred_title_embs)[0])
            current_title_irrelevance = 1 - np_max(self.cosine_similarity(current_title_emb, self.undesired_title_embs)[0])

            current_industry_emb = self.candidate_industries_embs[role_industry] .reshape(1, -1)       
            current_industry_score = np_max(self.cosine_similarity(current_industry_emb, self.preferred_industry_embs)[0])

            role_score = mean([desired_exp_score, nice_exp_score, undesired_exp_score, current_title_score, current_industry_score])
            scores.append(role_score), current_exp.append(is_current)
            
            if (current_title_score >= 0.8) or (current_industry_score >= 0.75) or (desired_exp_score >= 0.9):
#                 relevant_careers.append(f"Job role: {role_title} | Duration (in months): {role_duration}\nSummary: {role_description}")
                relevant_careers.append(f"Job role: {role_title} | Duration (in months): {role_duration}\nIndustry: {role_industry}")

        return mean(scores), relevant_careers

    def education_score(self, education, min_relevance=0.8):

        relevant_educations = []
        
        if education:
            last_end_year = 0
            for edu in education:
                if edu["end_year"] > last_end_year:
                    education_degree_emb = self.candidate_education_embs[f"{edu['degree']}, {edu['field_of_study']}"].reshape(1, -1)
                    education_score = np_max(self.cosine_similarity(education_degree_emb, self.preferred_education_embs)[0])
                    if education_score >= min_relevance:
                        relevant_educations.append(f"{edu['degree']}, {edu['field_of_study']}")
        else:
            education_score = 0

        return education_score, relevant_educations

    def skills_score(self, skills, min_relevance=0.9):

        def get_weight(param, value):
            references = {"endorsements": 200, "duration": 3}
            weight = 1 - exp(-value * log(10) / references[param])
            return weight

        relevant_skills = []
        relevant_primary_skills = {skill: 0 for skill in self.preferred_primary_skill_names}
        relevant_secondary_skills = {skill: 0 for skill in self.preferred_secondary_skill_names}

        skill_proficiency_weights = {"beginner": 0.25, "intermediate": 0.5, "advanced": 0.75, "expert": 1.}

        for skill in skills:

            name, proficiency = skill["name"], skill["proficiency"]
            duration, endorsements = round(skill["duration_months"] / 12, ndigits=1), skill["endorsements"]

            skill_emb = self.candidate_skill_embs[name].reshape(1, -1)
            primary_skill_scores = self.cosine_similarity(skill_emb, array(list(self.preferred_primary_skill_embs.values())))[0]
            secondary_skill_scores = self.cosine_similarity(skill_emb, array(list(self.preferred_secondary_skill_embs.values())))[0]

            primary_skill_score_idx, secondary_skill_score_idx = argmax(primary_skill_scores), argmax(secondary_skill_scores)

            matched_preferred_primary_skill = self.preferred_primary_skill_names[primary_skill_score_idx]
            matched_primary_skill_score = primary_skill_scores[primary_skill_score_idx]

            matched_preferred_secondary_skill = self.preferred_secondary_skill_names[secondary_skill_score_idx]
            matched_secondary_skill_score = secondary_skill_scores[secondary_skill_score_idx]
            
            if (matched_primary_skill_score >= min_relevance) or (matched_secondary_skill_score >= min_relevance):
#                 relevant_skills.append(f"{name} ({proficiency})")
                relevant_skills.append(name)

            if matched_primary_skill_score >= matched_secondary_skill_score:
                matched_primary_skill_score *= skill_proficiency_weights[proficiency] * get_weight("endorsements", endorsements)
                required_skill_duration = self.primary_skills_meta[matched_preferred_primary_skill]["min_months"] / 12
                matched_primary_skill_score = mean([matched_primary_skill_score, get_weight("duration", required_skill_duration)])
                if matched_primary_skill_score > relevant_primary_skills[matched_preferred_primary_skill]:
                    relevant_primary_skills[matched_preferred_primary_skill] = matched_primary_skill_score
            else:
                matched_secondary_skill_score *= skill_proficiency_weights[proficiency] * get_weight("endorsements", endorsements)
                required_skill_duration = self.secondary_skills_meta[matched_preferred_secondary_skill]["min_months"] / 12
                matched_secondary_skill_score = mean([matched_secondary_skill_score, get_weight("duration", required_skill_duration)])
                if matched_secondary_skill_score > relevant_secondary_skills[matched_preferred_secondary_skill]:
                    relevant_secondary_skills[matched_preferred_secondary_skill] = matched_secondary_skill_score

        existing_primary_skills = [value for value in relevant_primary_skills.values() if value > 0]
        existing_secondary_skills = [value for value in relevant_secondary_skills.values() if value > 0]

        number_of_required_skills_weight = len(existing_primary_skills) / len(self.preferred_primary_skill_names)
        number_of_optional_skills_weight = len(existing_secondary_skills) / len(self.preferred_secondary_skill_names)

        primary_skill_score = (mean(existing_primary_skills) * number_of_required_skills_weight) if existing_primary_skills else 0.
        secondary_skill_score = (mean(existing_secondary_skills) * number_of_optional_skills_weight) if existing_secondary_skills else 0.
        final_skill_score = mean([primary_skill_score, secondary_skill_score]) 

        return final_skill_score, relevant_skills

    def language_score(self, languages):

        language_weights = {"conversational": 0.5, "professional": 0.75, "native": 1.}

        scores = []
        for language in languages:
            name, proficiency = language["language"], language["proficiency"]
            if name in self.preferred_languages:
                scores.append(language_weights[proficiency])

        return mean(scores) if scores else 0.

    def redrob_signal_score(self, redrob_signals):

        def get_weight(param, value, proportionality):

            references = {
                "applications_submitted_30d": 22,
                "profile_views_received_30d": 374,
                "avg_response_time_hours": 279.9,
                "connection_count": 1852,
                "endorsements_received": 239,
                "saved_by_recruiters_30d": 76
            }

            if proportionality == "direct":
                weight = 1 - exp(-value * log(10) / references[param])
            elif proportionality == "inverse":
                weight = exp(-value * log(10) / references[param])
            else:
                weight = 0

            return weight

        unnormalized_signals = []
        normalized_signals = ["profile_completeness_score", "recruiter_response_rate", "github_activity_score", "interview_completion_rate", "offer_acceptance_rate"]

        scores = []
        for signal in normalized_signals:
            score = redrob_signals[signal]
            if score < 0: score = 0
            if signal in ["profile_completeness_score", "github_activity_score"]:
                scores.append(score / 100)
            else:
                scores.append(score)

        for signal in unnormalized_signals:
            proportionality = "inverse" if signal == "avg_response_time_hours" else "direct"
            scores.append(get_weight(signal, redrob_signals[signal], proportionality))

        scores.append(int(redrob_signals["open_to_work_flag"]))

        return mean(scores)
    
    def run_llm_inference(self, system_prompt, user_prompt, max_tokens=64, temperature=0.15, topP=0.99):

        response = self.llm_model.create_chat_completion(
            messages=[
                {"role": "system", "content": "\n".join([system_prompt, "/no_think"])},
                {"role": "user", "content": "\n".join([user_prompt, "/no_think"])}
            ],
            max_tokens=max_tokens, temperature=temperature, top_p=topP
        )
        response_text = response["choices"][0]["message"]["content"]
        
#         print('\n', response["usage"], '\n', response_text, '\n')
        
        return response_text
    
    def generate_reasons(self, profile_summary, relevant_educations, relevant_careers, relevant_skills):
        
        career_description = "; ".join([career for career in relevant_careers])
        education = "; ".join([degree for degree in relevant_educations])
        skills = ", ".join([skill for skill in relevant_skills])
        
        career_profile_details = ""
        if relevant_careers:
            career_profile_details += f"\n\nCareer Details:\n{career_description}"
        if relevant_educations:
            career_profile_details += f"\n\nEducation:\n{education}"
        if relevant_skills:
            career_profile_details += f"\n\nSkills Set:\n{skills}"
        if not career_profile_details:
            career_profile_details = f"Profile summary:\n{profile_summary}\n" + career_profile_details
        
        system_prompt = f"""
/no_think
Given a candidate profile details for a Senior AI Engineer role, provide a valid reason based on these details in a single short sentence of maximum 15 words only to consider the candidate profile for the job role.
/no_think
"""
        
        user_prompt = f"""
/no_think
Here are all the details of a candidate profile:
```
{career_profile_details}
```
Generate a reason to consider this profile by using < 15 words in plain text.
/no_think
"""
            
        ranking_reason = self.run_llm_inference(system_prompt, user_prompt)
        
        separator = "</think>"
        ranking_reason = ranking_reason[ranking_reason.find(separator)+len(separator):]
        
        return ranking_reason.strip().capitalize()        
    
    def rank_candidates(self, topK=100):
        
        main_start = time()
        start = main_start
        
        self.load_candidates_data()
        self.load_job_requirements()
        
        yoe_criteria = lambda x: (self.exp_min <= x <= self.exp_max)
        location_criteria = lambda country, city: (city in (self.primary_work_location + self.secondary_work_location)) or ((((country == "India") and (city not in (self.primary_work_location + self.secondary_work_location))) or (self.international_candidates_consideration and (country != "india"))) and data["redrob_signals"]["willing_to_relocate"])
        notice_period_criteria = lambda x: self.notice_period_min_days <= x <= self.notice_period_max_days
        work_mode_criteria = lambda x: (x != "remote")

        potential_candidates = []
        for data in tqdm(self.candidates_data, desc="Candidates initial filtering using Experience, Location, Notice Period & Preferred Work Mode: "):
            if yoe_criteria(data["profile"]["years_of_experience"]) and \
                location_criteria(data["profile"]["country"].capitalize(), data["profile"]["location"].capitalize()) and \
                notice_period_criteria(data["redrob_signals"]["notice_period_days"]) and \
                work_mode_criteria(data["redrob_signals"]["preferred_work_mode"]):
                    potential_candidates.append(data)

        print(f"\nOriginal Candidates Count: {len(self.candidates_data)} | Potential Candidates Count: {len(potential_candidates)}\n")
        
        self.prepare_embeddings(potential_candidates)
        
        scored_candidates = []
        scored_candidates_headers = ["candidate_id", "rank", "score", "reasoning"]

        for candidate in tqdm(potential_candidates, desc="Ranking Potential Candidates: "):

            candidate_id = candidate["candidate_id"]
            profile, career, education = candidate["profile"], candidate["career_history"], candidate["education"]
            skills, certifications, languages = candidate["skills"], candidate["certifications"], candidate["languages"]
            redrob_signals = candidate["redrob_signals"]

            pscore = self.profile_score(profile)
            lscore = self.language_score(languages)
            rscore = self.redrob_signal_score(redrob_signals)
            
            escore, relevant_educations = self.education_score(education)
            cscore, relevant_careers = self.career_score(career)
            sscore, relevant_skills = self.skills_score(skills)
            
            candidate_score = (0.2 * pscore) + (0.4 * cscore) + (0.2 * sscore) + (0.1 * escore) + (0.05 * lscore) + (0.05 * rscore)

            scored_candidates.append([candidate_id, relevant_educations, relevant_careers, relevant_skills, candidate_score])
            
        print()
        scored_candidates = sorted(scored_candidates, key=lambda x: float(x[-1]), reverse=True)[:topK]
        for ci, candidate in enumerate(tqdm(scored_candidates, desc="Generating Ranking Reasons: ")):
            candidate_id, relevant_educations, relevant_careers, relevant_skills, candidate_score = candidate
            ranking_reason = self.generate_reasons(profile["summary"], relevant_educations, relevant_careers, relevant_skills)
            scored_candidates[ci] = [candidate_id, ci+1, candidate_score, ranking_reason]

        scored_candidates_df = DataFrame(scored_candidates, columns=scored_candidates_headers)
        scored_candidates_df.to_csv("candidate_recommendations.csv", index=False)
        
        print(f"\nTotal candidates ranking time: {(time() - main_start):.2f} secs.\n")
        
if __name__ == "__main__":
    
    candidates_ranker = CandidateRankingSystem(
        candidates_data_file="./data/candidates.jsonl",
        job_requirements_file="./data/job_requirements.json",
        embedding_model_folder="./models/BGE-embed/BAAI/bge-small-en-v1.5",
        llm_model_file="./models/Qwen-GGUF/Qwen3-0.6B-Q4_K_M.gguf"
    )
    
    candidates_ranker.rank_candidates(topK=100)
    
    print(f"End to end processing time: {(time() - import_time):.2f} secs.\n")
    