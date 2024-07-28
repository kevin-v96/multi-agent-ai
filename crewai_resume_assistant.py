from crewai import Agent, Task, Crew
import os
from utils import get_openai_api_key, get_serper_api_key
from crewai_tools import FileReadTool, ScrapeWebsiteTool, MDXSearchTool, SerperDevTool

# bookkeeping
os.environ["OPENAI_API_KEY"] = get_openai_api_key()
os.environ["OPENAI_MODEL_NAME"] = "gpt-4o"
os.environ["SERPER_API_KEY"] = get_serper_api_key()

# import all the tools we'll need for this
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path="./kevin_resume.md")
semantic_search_resume = MDXSearchTool(mdx="./kevin_resume.md")

# Agents

# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on " "job posting to help job applicants",
    tools=[scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    ),
)

# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do increditble research on job applicants "
    "to help them stand out in the job market",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    ),
)

# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Engineers",
    goal="Find all the best ways to make a " "resume stand out in the job market.",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    ),
)

# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
    "based on the resume and job requirements",
    tools=[scrape_tool, search_tool, read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    ),
)

# Tasks
# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True,
)

# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the GitHub ({github_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True,
)

# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist,
)

# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer,
)

# crew
job_application_crew = Crew(
    agents=[researcher, profiler, resume_strategist, interview_preparer],
    tasks=[
        research_task,
        profile_task,
        resume_strategy_task,
        interview_preparation_task,
    ],
    verbose=True,
)

# Running the crew
job_application_inputs = {
    "job_posting_url": "https://depopcareers.com/careers/JR3536",
    "github_url": "https://github.com/kevin-v96",
    "personal_writeup": """Dear Hiring Manager,
I am excited to apply for the role of Data Scientist at *COMPANY NAME*. With a Master's degree in Data Science and two years of prior experience as a Product Engineer, I am confident that my skills and experience make me a perfect fit for this position.

As a Product Engineer, I have gained experience in developing interactive and user-friendly web applications. During my time working with React, I learned the importance of data visualisation and the role it plays in communicating complex information to users. This experience has given me a unique perspective on the importance of data and its analysis, which I am eager to bring to my work as a Data Scientist.

Throughout my Master's program, I have gained hands-on experience in data analysis, machine learning, and statistical modelling. I am proficient in Python and R, and have worked on projects involving data manipulation, cleaning, and visualisation using libraries such as Pandas, NumPy, and Matplotlib. In addition, I have experience working with SQL databases and have developed skills in using tools like Tableau and Excel for data analysis, visualisation and reporting.

My passion for AI and my experience as a frontend developer has given me a unique perspective on the role of data in user-centric applications. I am committed to using data to drive informed decisions and to provide the best possible user experience. I am also a team player and have excellent communication skills, which I believe are essential for any Data Scientist working in a collaborative and interdisciplinary environment.

I also have had the experience of working as an Intern Machine Learning Engineer at Flagright, a YC startup that tackles financial Fraud and AML. There, I helped set up their ML pipeline, including MLOps and its different steps on various platforms.

My Masters thesis and final project were on abstractive summarisation using SOTA LM models like LSTMs and Transformers. As such, I have a keen interest in LLMs and their usage. I also have certification as an AWS Solutions Architect and an Azure Data Scientist Associate.

I am excited about the opportunity to bring my skills and experience to *COMPANY NAME* and to make a meaningful contribution to your team. I look forward to the opportunity to further discuss my qualifications with you.

Sincerely,
Kevin Vegda
""",
}

### this execution will take a few minutes to run
result = job_application_crew.kickoff(inputs=job_application_inputs)
