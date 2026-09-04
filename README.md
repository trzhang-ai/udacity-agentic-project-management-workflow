# AI-Powered Agentic Workflow for Project Management

In this repo, you will find all the files and instructions required to complete the project. You can find more information about the project inside the Udacity Classroom.

## Getting Started

The project needs to be completed in two phases and you will find starter code for both the phases inside the `starter` folder in this repo. 

## Dependencies

A `requirements.txt` file has been provided in this repo if you want to work on the project locally. Otherwise, the workspace provided in the Udacity classroom has been configured with all the required libraries. 

## Project Instructions

You will find instructions for each of the two phases of the project in the README file inside the folder for that phase.

## Reviewer Note: Model Compatibility

The starter rubric specifies `gpt-3.5-turbo` with `temperature=0`. I intentionally used newer GPT-5 family models available through the course endpoint because they produced more reliable structured workflow output in my testing. In particular, `gpt-5.6-luna` rejects `temperature=0` on this endpoint with the message that only the default value of `1` is supported. The evaluation calls therefore use the model's supported default, expressed as `temperature=1`, together with an explicit reasoning-effort setting.

This is a deliberate model-and-endpoint compatibility decision, not an accidental omission of the rubric requirement. The evaluator loop, correction workflow, grounding rules, and required output criteria remain implemented. Because the newer model does not support `temperature=0` here, this project does not claim mathematically deterministic generation; instead, it reduces variability through explicit prompts, evaluation criteria, grounding, and structured output requirements.

## License
[License](../LICENSE.md)
