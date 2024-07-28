# multi-agent-ai
Some experimentation with the different multi-agent AI orchestration frameworks currently widely used.

Here are some of the ways to run this: After installing Poetry on your system, run `poetry install` to install all the pre-requisites.

## Weather bot using AutoGen

After that `poetry run python weather_app.py` will give you a local ip where the page is hosted and you can interact with the chatbot.

![alt text](images/image.png)

## Resume helper using CrewAI

`poetry run crewai_resume_assistant.py` after adding your resume in markdown format and adding the job posting link as well as a small writeup about yourself will help you alter your resume to match the job posting. Your new resume will be output as `tailored_resume.md`, and it will also output `interview_materials.md`, which will have questions you could prepare for to do well at interviews for this role.

## References
- [Creating a Multi-Agent Chatbot Using AutoGen: An End-to-End Guide](https://blog.arjun-g.com/creating-a-multi-agent-chatbot-using-autogen-an-end-to-end-guide-78b6671a96b4)
- [DeepLearning.ai - Multi AI Agent Systems with CrewAI](https://learn.deeplearning.ai/courses/multi-ai-agent-systems-with-crewai)
- [DeepLearning.ai - AI Agentic Design Patterns with AutoGen](https://learn.deeplearning.ai/courses/ai-agentic-design-patterns-with-autogen)
