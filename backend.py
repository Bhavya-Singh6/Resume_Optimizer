# backend.py
"""
Backend module for Resume Optimizer using CrewAI multi-agent system.
Handles job analysis, resume profiling, tailored resume creation, and interview prep.
"""
from dotenv import load_dotenv
import os
import time
import logging
import re
from typing import Dict, Optional, Tuple, Any

# -------------------------------
# Load environment
# -------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MAX_RESUME_CHARS = 22000
MAX_RETRY_ATTEMPTS = 3
BASE_RETRY_WAIT = 2.0

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    logger.warning("❌ GEMINI_API_KEY not found in environment. Set it in .env")

# -------------------------------
# Normalize model string
# -------------------------------
def normalize_model_name(raw_model: str) -> str:
    """Normalize the Gemini model name to the correct format."""
    if raw_model.startswith("models/"):
        raw_model = raw_model.replace("models/", "", 1)
    
    if "/" not in raw_model:
        return f"gemini/{raw_model}"
    
    provider, name = raw_model.split("/", 1)
    return raw_model if provider != "models" else f"gemini/{name}"

_raw_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
GEMINI_MODEL = normalize_model_name(_raw_model)
logger.info(f"Normalized GEMINI_MODEL -> {GEMINI_MODEL}")

# -------------------------------
# Initialize LLM
# -------------------------------
llm = None
used_llm_backend = None

try:
    from crewai import LLM as CrewLLM
    try:
        llm = CrewLLM(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)
        used_llm_backend = "crewai.LLM"
        logging.info("Using crewai.LLM for Gemini integration.")
    except Exception as e:
        logging.warning(f"crewai.LLM initialization failed: {e}")
        llm = None
except Exception:
    logging.info("crewai.LLM not available; will try langchain-google-genai fallback.")
    llm = None

if llm is None:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model_only = GEMINI_MODEL.split("/", 1)[-1]  # remove provider prefix
        llm = ChatGoogleGenerativeAI(
            model=model_only,
            google_api_key=GEMINI_API_KEY,
            convert_system_message_to_human=True
        )
        used_llm_backend = "langchain_google_genai"
        logging.info("Using langchain_google_genai.ChatGoogleGenerativeAI for Gemini integration.")
    except Exception as e:
        logging.exception("LangChain fallback failed")
        llm = None

if llm is None:
    logging.error("No usable LLM backend available. CrewAI agents will not run.")

# -------------------------------
# Import Crew classes
# -------------------------------
try:
    from crewai import Agent, Task, Crew
except Exception as e:
    logging.exception("Failed to import crewai. Agents will not run.")
    Agent = Task = Crew = None

# -------------------------------
# Create Agents
# -------------------------------
def make_agent(role: str, goal: str, backstory: str):
    """
    Create a CrewAI Agent with the specified role, goal, and backstory.
    
    Args:
        role: The role of the agent
        goal: The goal of the agent
        backstory: The backstory/context for the agent
        
    Returns:
        Agent configured with the specified parameters and LLM
    """
    if Agent is None:
        return None
    kwargs = dict(role=role, goal=goal, backstory=backstory, verbose=True)
    if llm:
        kwargs["llm"] = llm
    return Agent(**kwargs)

researcher = make_agent(
    role="Job Researcher",
    goal="Extract and structure all requirements and details from job descriptions",
    backstory="""I am an expert job posting analyzer with years of experience. I specialize in:
    - Identifying both explicit and implicit requirements
    - Breaking down technical and soft skill requirements
    - Understanding company culture and work environment
    - Recognizing key responsibilities and success factors"""
)

profiler = make_agent(
    role="Profile Analyzer",
    goal="Create detailed analysis of candidate qualifications and experience",
    backstory="""I am a professional resume analyst with expertise in:
    - Evaluating technical and soft skills from experience
    - Identifying key achievements and metrics
    - Understanding career progression and growth
    - Spotting relevant certifications and training"""
)

strategist = make_agent(
    role="Resume Strategist",
    goal="Create targeted, ATS-friendly resumes that highlight relevant qualifications",
    backstory="""I am a resume optimization specialist focusing on:
    - Matching candidate skills to job requirements
    - Writing compelling achievement statements
    - Using industry-specific keywords
    - Creating ATS-optimized formatting"""
)

interview_coach = make_agent(
    role="Interview Coach",
    goal="Prepare comprehensive interview strategies and responses",
    backstory="""I am an experienced interview preparation expert who:
    - Creates targeted interview questions and answers
    - Develops STAR format response templates
    - Identifies key talking points from experience
    - Prepares strategies for challenging questions"""
)

# -------------------------------
# Create Tasks
# -------------------------------
def make_task(description, expected_output, agent, output_key, input_keys=None):
    if Task is None:
        return None
    kwargs = dict(
        description=description,
        expected_output=expected_output,
        agent=agent,
        output_key=output_key
    )
    if input_keys:
        kwargs["context"] = input_keys  # CrewAI uses 'context' instead of 'input_keys'
    return Task(**kwargs)

research_task = Task(
    description="""Analyze the job description and create a structured markdown report including:
    1. Technical Skills Required
       - List all technical skills, tools, and technologies
       - Note which are required vs preferred
    2. Soft Skills Required
       - Communication requirements
       - Team collaboration aspects
       - Other interpersonal skills
    3. Main Responsibilities
       - Day-to-day duties
       - Project responsibilities
       - Team interactions
    4. Experience & Education
       - Years of experience required
       - Education requirements
       - Industry-specific requirements
    5. Company Details
       - Company description
       - Work environment
       - Benefits and perks
       
    Job Description: {job_desc}""",
    expected_output="""A detailed markdown-formatted analysis with ALL of these sections:
    # Job Analysis
    ## Technical Requirements
    [List of skills...]
    ## Soft Skills
    [List of skills...]
    ## Key Responsibilities
    [List of duties...]
    ## Requirements
    [Experience/education...]
    ## Company Information
    [Company details...]""",
    agent=researcher,
    async_execution=False,
    output_pydantic=None,
    output_json=None,
    output_file=None
)

profile_task = Task(
    description="""Analyze the resume and create a structured markdown report including:
    1. Technical Skills
       - List all technical skills found
       - Note experience level with each
    2. Soft Skills
       - Communication abilities
       - Leadership experience
       - Team collaboration
    3. Work Experience
       - Key achievements
       - Responsibilities held
       - Technologies used
    4. Education & Certifications
       - Degrees and certifications
       - Relevant training
    5. Overall Profile
       - Career progression
       - Areas of expertise
       
    Resume Text: {resume_text}""",
    expected_output="""A detailed markdown-formatted analysis with ALL of these sections:
    # Candidate Profile
    ## Technical Skills
    [List of skills...]
    ## Soft Skills
    [List of skills...]
    ## Experience Highlights
    [Key achievements...]
    ## Education & Training
    [Details...]
    ## Professional Summary
    [Overview...]""",
    agent=profiler,
    async_execution=False
)

strategy_task = Task(
    description="""Create a tailored resume by:
    1. Compare job requirements with candidate profile
    2. Prioritize matching skills and experiences
    3. Rewrite experience points to highlight relevant achievements
    4. Include key terms from job description
    5. Format in a clear, ATS-friendly structure
    
    The resume must include:
    - Professional Summary
    - Skills Section
    - Work Experience
    - Education
    - Additional Sections (if relevant)
    
    Use the context from previous tasks for job analysis and candidate profile.""",
    expected_output="""A complete, formatted resume with ALL of these sections:
    # Professional Summary
    [Tailored summary...]
    # Skills
    [Relevant skills...]
    # Work Experience
    [Tailored experience...]
    # Education
    [Education details...]""",
    agent=strategist,
    async_execution=False,
    context=[research_task, profile_task]
)

interview_task = Task(
    description="""Create an interview preparation guide including:
    1. Technical Questions
       - Based on required skills
       - With suggested answers
    2. Behavioral Questions
       - Using STAR format
       - Based on job responsibilities
    3. Achievement Discussion
       - Key points from resume
       - Metrics and results
    4. Company Research
       - Culture fit questions
       - Questions to ask interviewer
       
    Use the context from previous tasks for job analysis and tailored resume.""",
    expected_output="""A comprehensive interview guide with ALL of these sections:
    # Interview Preparation Guide
    ## Technical Questions
    [Questions and answers...]
    ## Behavioral Questions
    [STAR format questions...]
    ## Key Talking Points
    [Achievement discussions...]
    ## Questions to Ask
    [Research-based questions...]""",
    agent=interview_coach,
    async_execution=False,
    context=[research_task, strategy_task]
)

# -------------------------------
# Initialize Crew
# -------------------------------
crew = None
if Crew:
    agents = [a for a in (researcher, profiler, strategist, interview_coach) if a]
    tasks = [t for t in (research_task, profile_task, strategy_task, interview_task) if t]
    if agents and tasks:
        crew = Crew(agents=agents, tasks=tasks, verbose=True)
    else:
        logging.error("Crew cannot start: missing agents or tasks")

# -------------------------------
# Resume section extraction
# -------------------------------
SECTION_HEADERS_RE = re.compile(
    r"(?im)^(summary|professional summary|skills|technical skills|experience|work experience|projects|education|certifications|achievements)[:\s]*$",
    re.MULTILINE
)

def extract_key_resume_sections(text: str, max_chars: int = MAX_RESUME_CHARS) -> Tuple[str, bool]:
    """
    Extract key sections from resume text and truncate to prevent token overflow.
    
    Args:
        text: The resume text to process
        max_chars: Maximum character limit
        
    Returns:
        Tuple of (processed_text, was_truncated)
    """
    matches = list(SECTION_HEADERS_RE.finditer(text))
    if matches:
        parts = []
        indices = [m.start() for m in matches] + [len(text)]
        for i, m in enumerate(matches):
            chunk = text[m.start():indices[i + 1]].strip()
            if chunk:
                parts.append(chunk)
        combined = "\n\n".join(parts)
        return (combined[:max_chars], True) if len(combined) > max_chars else (combined, False)
    return (text[:max_chars], True) if len(text) > max_chars else (text, False)

# -------------------------------
# Kickoff with retries
# -------------------------------
def kickoff_with_retries(inputs: Dict[str, str], max_attempts: int = MAX_RETRY_ATTEMPTS, base_wait: float = BASE_RETRY_WAIT) -> Dict[str, str]:
    """
    Execute the crew with retry logic.
    
    Args:
        inputs: Dictionary containing 'job_desc' and 'resume_text'
        max_attempts: Maximum number of retry attempts
        base_wait: Base wait time for exponential backoff
        
    Returns:
        Dictionary with job_analysis, profile_analysis, tailored_resume, and interview_prep
        
    Raises:
        RuntimeError: If crew/agents not initialized or all attempts fail
    """
    if not crew or not llm or not all([researcher, profiler, strategist, interview_coach]):
        raise RuntimeError("Crew/Agents/LLM not initialized")

    logger.info("Starting crew execution with inputs:")
    logger.info(f"- Job description length: {len(inputs.get('job_desc', ''))} chars")
    logger.info(f"- Resume length: {len(inputs.get('resume_text', ''))} chars")

    attempt = 0
    while attempt < max_attempts:
        attempt += 1
        try:
            logging.info(f"Attempt {attempt}/{max_attempts}")
            
            # Run the crew
            result = crew.kickoff(inputs=inputs)
            
            logging.info(f"Result type: {type(result)}")
            logging.info(f"Result: {result}")
            
            # CrewAI returns a CrewOutput object with tasks_output attribute
            if hasattr(result, 'tasks_output'):
                tasks_output = result.tasks_output
                logging.info(f"Found {len(tasks_output)} task outputs")
                
                # Extract outputs from each task
                outputs = {}
                if len(tasks_output) >= 4:
                    outputs['job_analysis'] = str(tasks_output[0].raw) if hasattr(tasks_output[0], 'raw') else str(tasks_output[0])
                    outputs['profile_analysis'] = str(tasks_output[1].raw) if hasattr(tasks_output[1], 'raw') else str(tasks_output[1])
                    outputs['tailored_resume'] = str(tasks_output[2].raw) if hasattr(tasks_output[2], 'raw') else str(tasks_output[2])
                    outputs['interview_prep'] = str(tasks_output[3].raw) if hasattr(tasks_output[3], 'raw') else str(tasks_output[3])
                else:
                    raise RuntimeError(f"Expected 4 task outputs, got {len(tasks_output)}")
                    
                logging.info("Successfully extracted all task outputs")
                return outputs
            elif hasattr(result, 'raw'):
                # Single output
                logging.warning("Got single output instead of multiple task outputs")
                return {"error": "Unexpected result structure - got single output"}
            else:
                logging.error(f"Unknown result structure: {dir(result)}")
                raise RuntimeError("Unknown result structure from crew")
                
        except Exception as e:
            msg = str(e).lower()
            if any(x in msg for x in ["rate limit", "timeout", "503", "500"]):
                wait = base_wait * (2 ** (attempt - 1))
                logging.warning(f"Transient error: {e}, retrying in {wait}s")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Crew failed after {max_attempts} attempts")

# -------------------------------
# Main backend
# -------------------------------
def run_backend(job_desc: str, resume_text: str) -> Dict[str, Any]:
    """
    Main entry point for resume optimization backend.
    
    Args:
        job_desc: Job description text
        resume_text: Resume text content
        
    Returns:
        Dictionary containing job_analysis, profile_analysis, tailored_resume, 
        interview_prep, and debug information. Returns error dict on failure.
    """
    try:
        if not GEMINI_API_KEY:
            return {"error": "GEMINI_API_KEY missing"}
        if not crew:
            return {"error": "Crew not initialized; check agent/LLM setup"}

        job_desc, resume_text = job_desc.strip(), resume_text.strip()
        safe_resume, was_truncated = extract_key_resume_sections(resume_text)
        if was_truncated:
            safe_resume = "(Truncated resume for token safety)\n\n" + safe_resume

        inputs = {"job_desc": job_desc, "resume_text": safe_resume}

        # kickoff_with_retries now returns a dict with the outputs already
        outputs = kickoff_with_retries(inputs)
        
        # outputs is already a dict with job_analysis, profile_analysis, tailored_resume, interview_prep

        debug_info = {
            "used_llm_backend": used_llm_backend,
            "GEMINI_MODEL": GEMINI_MODEL,
            "llm_configured": llm is not None,
            "crew_repr": repr(crew)[:800] if crew else "Crew not initialized",
        }

        outputs["debug"] = debug_info
        return outputs

    except Exception as e:
        logging.exception("run_backend failed")
        return {"error": f"{type(e).__name__}: {str(e)}"}
