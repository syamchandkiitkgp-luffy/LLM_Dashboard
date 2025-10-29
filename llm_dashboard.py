"""
ServiceNow Strategic Planning & Analytics Dashboard
Complete Application with Data Generation and Visualization
Includes Month-over-Month tracking for all KPIs
With centered text in KPI metric boxes
Plus Agentic AI-powered chatbot using Google Gemini API
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
import os
import json
import traceback
from google import genai
import time

# ============================================================================
# DATA DICTIONARY & METADATA
# ============================================================================

DATA_DICTIONARY = {
    "dataset_info": {
        "name": "ServiceNow Strategic Planning & Analytics Dataset",
        "description": "Comprehensive dataset tracking customer metrics, revenue, pipeline, renewals, and partner performance for ServiceNow",
        "purpose": "Enable strategic planning and analytics through KPI tracking, trend analysis, and data-driven decision making",
        "granularity": "Monthly data per client",
        "date_range": "January 2023 to October 2025",
        "total_clients": 20,
        "total_records": "~440 records (20 clients × ~34 months)"
    },
    "columns": {
        # Time & Client Identifiers
        "month": {
            "type": "datetime",
            "description": "Month-year timestamp for the record",
            "granularity": "Monthly",
            "example": "2025-10-01"
        },
        "client_name": {
            "type": "string",
            "description": "Unique identifier for each client",
            "values": "Client_A through Client_T",
            "example": "Client_A"
        },

        # Client Attributes
        "industry": {
            "type": "categorical",
            "description": "Industry sector of the client",
            "values": ["Healthcare", "Financial Services", "Technology", "Retail", "Manufacturing"],
            "use_case": "Segment analysis by industry"
        },
        "tier": {
            "type": "categorical",
            "description": "Customer tier based on size and revenue",
            "values": ["Enterprise", "Mid-Market", "SMB"],
            "use_case": "Segment analysis by customer size"
        },
        "acquisition_source": {
            "type": "categorical",
            "description": "Channel through which customer was acquired",
            "values": ["Direct Sales", "Channel Partner", "Alliance Partner", "Inside Sales"],
            "use_case": "Channel performance analysis"
        },

        # Revenue Metrics
        "mrr": {
            "type": "numeric",
            "description": "Monthly Recurring Revenue in USD",
            "unit": "USD",
            "calculation": "Monthly subscription revenue",
            "use_case": "Track monthly revenue performance"
        },
        "arr": {
            "type": "numeric",
            "description": "Annual Recurring Revenue in USD",
            "unit": "USD",
            "calculation": "MRR × 12",
            "use_case": "Track annual revenue projection"
        },
        "expansion_revenue": {
            "type": "numeric",
            "description": "Additional revenue from upsells/cross-sells",
            "unit": "USD",
            "use_case": "Track growth within existing accounts"
        },

        # Pipeline Metrics
        "pipeline_value": {
            "type": "numeric",
            "description": "Total value of open sales opportunities",
            "unit": "USD",
            "use_case": "Forecast future revenue"
        },
        "weighted_pipeline": {
            "type": "numeric",
            "description": "Pipeline value adjusted by win probability",
            "unit": "USD",
            "calculation": "pipeline_value × win_rate",
            "use_case": "Realistic revenue forecasting"
        },
        "win_rate": {
            "type": "numeric",
            "description": "Probability of winning deals",
            "unit": "Decimal (0-1)",
            "range": "0.15 to 0.45",
            "use_case": "Sales effectiveness measurement"
        },
        "pipeline_coverage": {
            "type": "numeric",
            "description": "Pipeline to quota ratio",
            "unit": "Ratio",
            "calculation": "pipeline_value / quarterly_quota",
            "use_case": "Assess pipeline health"
        },

        # Customer Health & Engagement
        "customer_health_score": {
            "type": "numeric",
            "description": "Overall customer health indicator",
            "unit": "Score (0-100)",
            "range": "60-95",
            "use_case": "Identify at-risk customers"
        },
        "nps_score": {
            "type": "numeric",
            "description": "Net Promoter Score",
            "unit": "Score (-100 to 100)",
            "range": "30-80",
            "use_case": "Measure customer loyalty"
        },

        # User Metrics
        "active_users": {
            "type": "numeric",
            "description": "Total number of active users",
            "unit": "Count",
            "use_case": "Track product adoption"
        },

        # Renewal Metrics
        "is_renewal_period": {
            "type": "boolean",
            "description": "Whether customer is in renewal period (within 90 days)",
            "values": [True, False],
            "use_case": "Identify renewal opportunities"
        },
        "days_to_renewal": {
            "type": "numeric",
            "description": "Days until contract renewal",
            "unit": "Days",
            "use_case": "Prioritize renewal activities"
        },
        "renewal_risk_score": {
            "type": "numeric",
            "description": "Risk of customer not renewing",
            "unit": "Score (0-1)",
            "range": "0-1 (higher = more risk)",
            "use_case": "Identify at-risk renewals"
        },
        "is_churned": {
            "type": "boolean",
            "description": "Whether customer has churned",
            "values": [0, 1],
            "use_case": "Track customer loss"
        },

        # Partner Metrics
        "partner_name": {
            "type": "string",
            "description": "Name of partner (if applicable)",
            "example": "Partner_1",
            "use_case": "Track partner attribution"
        },
        "partner_influenced_revenue": {
            "type": "numeric",
            "description": "Revenue attributed to partners",
            "unit": "USD",
            "use_case": "Measure partner contribution"
        },
        "partner_engagement_score": {
            "type": "numeric",
            "description": "Partner relationship health",
            "unit": "Score (0-100)",
            "range": "60-95",
            "use_case": "Manage partner relationships"
        }
    }
}

# ============================================================================
# ENHANCED AGENTIC AI SYSTEM WITH OUT-OF-CONTEXT HANDLING (2025)
# ============================================================================

import time

class ContextValidator:
    """Validates if questions are within the dashboard data context"""

    def __init__(self, api_key, data_dictionary):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.data_dictionary = data_dictionary

    def validate_context(self, user_question):
        """Determine if question is within data context or general knowledge"""

        if not self.client:
            return {"in_context": True, "confidence": 0.5, "reasoning": "No API key"}

        # Get available metrics from data dictionary
        available_metrics = list(self.data_dictionary.get("columns", {}).keys())

        prompt = f"""You are a context validator for a ServiceNow analytics dashboard.

AVAILABLE DATA CONTEXT:
Dataset: {self.data_dictionary['dataset_info']['name']}
Available Metrics: {', '.join(available_metrics[:30])}
Time Period: {self.data_dictionary['dataset_info']['date_range']}
Granularity: {self.data_dictionary['dataset_info']['granularity']}

USER QUESTION: "{user_question}"

TASK: Determine if this question can be answered using the available dashboard data.

CLASSIFICATION RULES:
1. IN-CONTEXT (answer with data):
   - Questions about metrics in the available list
   - Questions about trends, comparisons, aggregations
   - Questions about specific clients, industries, time periods
   - Examples: "What's the total MRR?", "Show top 10 clients", "MRR trend"

2. OUT-OF-CONTEXT (answer with general knowledge):
   - General questions about business concepts
   - Questions about methodologies or best practices
   - "What is..." or "How to..." questions not requiring data
   - Examples: "What is MRR?", "How to improve churn?", "What is ServiceNow?"

3. HYBRID (needs both data and knowledge):
   - Requests for insights with explanations
   - Examples: "Why is churn high and how to fix it?"

OUTPUT FORMAT (JSON only):
{{
    "classification": "in_context" or "out_of_context" or "hybrid",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "suggested_approach": "data_analysis" or "knowledge_response" or "combined"
}}

RESPOND WITH JSON ONLY:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config={"temperature": 0.2}
            )

            # Parse response
            response_text = response.text
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "{" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
            else:
                json_str = response_text

            result = json.loads(json_str)
            result.setdefault("classification", "in_context")
            result.setdefault("confidence", 0.7)

            return result

        except Exception as e:
            return {
                "classification": "in_context",
                "confidence": 0.5,
                "reasoning": f"Error in validation: {str(e)}",
                "suggested_approach": "data_analysis"
            }


class GeneralKnowledgeAgent:
    """Handles out-of-context questions using general knowledge"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = genai.Client(api_key=api_key) if api_key else None

    def answer_general_question(self, user_question, data_context=None):
        """Answer general knowledge questions"""

        if not self.client:
            return "Unable to answer - API key required"

        context_info = ""
        if data_context:
            context_info = f"""

DASHBOARD CONTEXT (for reference):
The user is viewing a ServiceNow Strategic Planning & Analytics Dashboard with:
- Revenue metrics (MRR, ARR, Pipeline)
- Customer health metrics (NPS, Health Score, Churn)
- Renewal metrics (Days to renewal, Risk scores)
- Partner metrics (Revenue, Engagement)

You can reference this context if relevant to the question."""

        prompt = f"""You are a senior business consultant and ServiceNow expert.

USER QUESTION: "{user_question}"
{context_info}

INSTRUCTIONS:
1. Provide a clear, professional answer
2. Use business language (not technical jargon)
3. Structure your response with headers if needed
4. Include practical examples when helpful
5. If the question relates to metrics, explain what they are and why they matter
6. Keep it concise but comprehensive (200-400 words)

FORMAT YOUR RESPONSE:
- Use ## for main headers
- Use bullet points for lists
- Bold key terms with **term**
- Provide actionable insights

YOUR RESPONSE:"""

        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=prompt,
                config={"temperature": 0.4, "max_output_tokens": 1000}
            )
            return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"


class EnhancedBaseAgent:
    """Enhanced base agent with retry logic and performance tracking"""

    def __init__(self, api_key, role, instructions):
        self.api_key = api_key
        self.role = role
        self.instructions = instructions
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.metrics = {"total_calls": 0, "successful_calls": 0, "errors": []}

    def query(self, prompt, temperature=0.3, max_retries=2):
        """Query with retry logic and exponential backoff"""

        if not self.client:
            return "Error: API key not provided"

        for attempt in range(max_retries + 1):
            try:
                self.metrics["total_calls"] += 1

                full_prompt = f"""ROLE: {self.role}

{self.instructions}

TASK:
{prompt}

YOUR RESPONSE:"""

                response = self.client.models.generate_content(
                    model="gemini-2.0-flash-exp",
                    contents=full_prompt,
                    config={
                        "temperature": temperature,
                        "top_p": 0.95,
                        "top_k": 40,
                        "max_output_tokens": 2048
                    }
                )

                self.metrics["successful_calls"] += 1
                return response.text

            except Exception as e:
                if attempt < max_retries:
                    time.sleep(2 ** attempt)
                else:
                    self.metrics["errors"].append(str(e))
                    return f"Error: {str(e)}"

        return "Error: Max retries exceeded"


class EnhancedOrchestratorAgent(EnhancedBaseAgent):
    """Orchestrator with context awareness"""

    def __init__(self, api_key):
        role = "Strategic Orchestrator"
        instructions = """Analyze questions and plan optimal execution strategy.

DECISION FRAMEWORK:
- Data queries → Python coding
- Visualizations → Plotting  
- Insights → Summarizing
- Complex analysis → Multiple agents"""
        super().__init__(api_key, role, instructions)

    def plan(self, user_question, df):
        """Create execution plan"""

        prompt = f"""Analyze: "{user_question}"

Available data: {len(df)} rows, Columns: {', '.join(df.columns[:15].tolist())}

Determine needs:
- Data processing? (aggregations, filtering)
- Visualization? (charts, graphs)
- Summary? (insights, recommendations)

OUTPUT JSON:
{{"needs_data": true/false, "needs_viz": true/false, "needs_summary": true/false, "complexity": "low/medium/high"}}"""

        response = self.query(prompt, temperature=0.2)

        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response

            plan = json.loads(json_str)
            plan.setdefault("needs_data", True)
            plan.setdefault("needs_viz", False)
            plan.setdefault("needs_summary", True)

            return plan
        except:
            return {"needs_data": True, "needs_viz": False, "needs_summary": True}


class EnhancedPythonCodingAgent(EnhancedBaseAgent):
    """Advanced coding with best practices"""

    def __init__(self, api_key):
        role = "Expert Python Data Analyst"
        instructions = """Write clean pandas code following best practices.

RULES:
✓ Input: df, Output: result_df
✓ Handle nulls with .dropna() or .fillna()
✓ Reset index after groupby
✓ Use meaningful names
✗ No print statements
✗ No comments

PATTERNS:
# Aggregation
result_df = df[df['val']>0].groupby('cat')['val'].sum().reset_index()

# Time series
result_df = df.groupby('month')['metric'].sum().reset_index().sort_values('month')

# Top N
result_df = df.nlargest(10, 'value')[['name', 'value']]"""
        super().__init__(api_key, role, instructions)

    def generate_code(self, task, df):
        """Generate pandas code"""

        prompt = f"""Task: {task}

Columns: {', '.join(df.columns.tolist())}
Shape: {df.shape}

Write pandas code (no markdown, no explanations):"""

        code = self.query(prompt, temperature=0.1)

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code


class EnhancedPythonReviewAgent(EnhancedBaseAgent):
    """Code reviewer with error patterns"""

    def __init__(self, api_key):
        role = "Senior Code Reviewer"
        instructions = """Fix pandas code errors.

ERROR PATTERNS:
1. NameError → Check column names
2. KeyError → Add .reset_index()
3. ValueError → Handle empty data
4. TypeError → Convert data types

OUTPUT: Fixed code only."""
        super().__init__(api_key, role, instructions)

    def fix_code(self, code, error, columns):
        """Fix broken code"""

        prompt = f"""Fix this code:

{code}

Error: {error}
Columns: {', '.join(columns)}

Return fixed code only:"""

        fixed = self.query(prompt, temperature=0.1)

        if "```python" in fixed:
            fixed = fixed.split("```python")[1].split("```")[0].strip()
        elif "```" in fixed:
            fixed = fixed.split("```")[1].split("```")[0].strip()

        return fixed


class EnhancedPlottingAgent(EnhancedBaseAgent):
    """Plotting with null safety"""

    def __init__(self, api_key):
        role = "Data Visualization Expert"
        instructions = """Create Plotly visualizations.

CHART TYPES:
- Time series → LINE (px.line)
- Categories → BAR (px.bar)
- Part-whole → PIE (px.pie)
- Distribution → HISTOGRAM

CRITICAL:
1. Remove nulls: result_df = result_df.dropna()
2. Check empty: if len(result_df) == 0
3. Variable: fig
4. Height: 400-600px

TEMPLATE:
result_df = result_df.dropna()
if len(result_df) > 0:
    fig = px.line(result_df, x='x', y='y', title='Title')
else:
    fig = go.Figure()
fig.update_layout(height=500)"""
        super().__init__(api_key, role, instructions)

    def generate_visualization(self, task, result_df):
        """Generate viz code"""

        df_info = f"Columns: {result_df.columns.tolist()}, Shape: {result_df.shape}"

        prompt = f"""Create chart for: {task}

Data: {df_info}

Return code only (with null handling):"""

        code = self.query(prompt, temperature=0.2)

        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code


class EnhancedSummarizingAgent(EnhancedBaseAgent):
    """Executive insights generator"""

    def __init__(self, api_key):
        role = "Senior Business Consultant"
        instructions = """Provide executive-level insights.

STRUCTURE:
## Executive Summary (2-3 sentences)
## Key Insights (3-5 bullets with numbers)
## Recommendations (2-3 actionable items)

STYLE:
✓ Business language
✓ Specific numbers
✓ Actionable advice
✗ No technical jargon"""
        super().__init__(api_key, role, instructions)

    def generate_summary(self, task, result_df):
        """Generate insights"""

        data_summary = result_df.head(15).to_string() if isinstance(result_df, pd.DataFrame) else str(result_df)

        prompt = f"""Question: {task}

Data:
{data_summary}

Provide executive insights:"""

        return self.query(prompt, temperature=0.4)


class EnhancedAgenticChatbot:
    """Enhanced chatbot with out-of-context handling"""

    def __init__(self, api_key, df, data_dictionary):
        self.api_key = api_key
        self.df = df
        self.data_dictionary = data_dictionary

        # Initialize all agents
        self.context_validator = ContextValidator(api_key, data_dictionary)
        self.knowledge_agent = GeneralKnowledgeAgent(api_key)
        self.orchestrator = EnhancedOrchestratorAgent(api_key)
        self.coder = EnhancedPythonCodingAgent(api_key)
        self.reviewer = EnhancedPythonReviewAgent(api_key)
        self.plotter = EnhancedPlottingAgent(api_key)
        self.summarizer = EnhancedSummarizingAgent(api_key)

    def execute(self, user_question):
        """Main execution with context validation"""
        start_time = datetime.now()

        try:
            # Step 1: Validate context
            validation = self.context_validator.validate_context(user_question)

            result = {
                "question": user_question,
                "context_validation": validation,
                "timestamp": start_time.isoformat()
            }

            # Step 2: Route based on context
            if validation["classification"] == "out_of_context":
                # Answer using general knowledge
                answer = self.knowledge_agent.answer_general_question(
                    user_question,
                    data_context=self.data_dictionary['dataset_info']
                )
                result["answer_type"] = "general_knowledge"
                result["summary"] = answer
                result["result_df"] = None

            elif validation["classification"] == "hybrid":
                # Combine data analysis with knowledge
                data_result = self._process_data_question(user_question)
                knowledge_answer = self.knowledge_agent.answer_general_question(
                    user_question,
                    data_context=self.data_dictionary['dataset_info']
                )
                result.update(data_result)
                result["additional_context"] = knowledge_answer
                result["answer_type"] = "hybrid"

            else:
                # Process as data question
                data_result = self._process_data_question(user_question)
                result.update(data_result)
                result["answer_type"] = "data_analysis"

            # Add execution time
            result["execution_time"] = (datetime.now() - start_time).total_seconds()

            return result

        except Exception as e:
            return {
                "question": user_question,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "result_df": self.df.head(10)
            }

    def _process_data_question(self, user_question):
        """Process data-related questions"""

        result = {}

        # Orchestration
        plan = self.orchestrator.plan(user_question, self.df)
        result["plan"] = plan

        # Coding & Execution
        if plan.get("needs_data", True):
            code = self.coder.generate_code(user_question, self.df)
            result["code"] = code

            exec_result = self._safe_execute(code)
            result.update(exec_result)

        # Visualization
        if plan.get("needs_viz", False) and result.get("result_df") is not None:
            viz_code = self.plotter.generate_visualization(user_question, result["result_df"])
            result["viz_code"] = viz_code

            viz_result = self._safe_plot(viz_code, result["result_df"])
            result.update(viz_result)

        # Summary
        if plan.get("needs_summary", True) and result.get("result_df") is not None:
            summary = self.summarizer.generate_summary(user_question, result["result_df"])
            result["summary"] = summary

        return result

    def _safe_execute(self, code, max_retries=2):
        """Execute code with retry logic"""

        for attempt in range(max_retries + 1):
            try:
                local_vars = {"df": self.df.copy(), "pd": pd, "np": np}
                exec(code, {}, local_vars)
                result_df = local_vars.get("result_df")

                if result_df is None:
                    raise ValueError("No result_df variable created")

                # Handle nulls
                if isinstance(result_df, pd.DataFrame):
                    result_df = result_df.dropna(how='all')
                    for col in result_df.columns:
                        if result_df[col].dtype in ['float64', 'int64']:
                            result_df[col] = result_df[col].fillna(0)

                return {"result_df": result_df, "execution_status": "success", "attempts": attempt + 1}

            except Exception as e:
                if attempt < max_retries:
                    # Try to fix the code
                    code = self.reviewer.fix_code(code, str(e), self.df.columns.tolist())
                else:
                    return {"result_df": self.df.head(10), "error": str(e), "execution_status": "error"}

        return {"result_df": self.df.head(10), "error": "Max retries exceeded"}

    def _safe_plot(self, viz_code, result_df):
        """Create visualization safely"""
        try:
            local_vars = {"result_df": result_df.copy(), "px": px, "go": go, "pd": pd}
            exec(viz_code, {}, local_vars)
            fig = local_vars.get("fig")
            return {"chart": fig}
        except Exception as e:
            return {"chart": None, "viz_error": str(e)}


# ============================================================================
# DATA GENERATION SECTION
# ============================================================================

def generate_dataset():
    """Generate the complete dataset for the dashboard"""

    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    # Generate 20 clients
    num_clients = 20
    client_names = [f"Client_{chr(65+i)}" if i < 26 else f"Client_{i}" for i in range(num_clients)]

    # Generate months from Jan 2023 to October 2025 (34 months)
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2025, 10, 1)
    months = pd.date_range(start=start_date, end=end_date, freq='MS')

    # Initialize lists to store data
    data = []

    for client in client_names:
        # Client-specific attributes
        client_start_month = random.choice(months[:24])
        industry = random.choice(['Healthcare', 'Financial Services', 'Technology', 'Retail', 'Manufacturing'])
        tier = random.choice(['Enterprise', 'Mid-Market', 'SMB'])
        acquisition_source = random.choice(['Direct Sales', 'Channel Partner', 'Alliance Partner', 'Inside Sales'])

        # Partner information (if applicable)
        partner_name = f"Partner_{random.randint(1, 10)}" if acquisition_source in ['Channel Partner', 'Alliance Partner'] else None

        # Initial values
        initial_mrr = random.randint(5000, 50000) if tier == 'Enterprise' else random.randint(2000, 15000) if tier == 'Mid-Market' else random.randint(500, 5000)
        initial_users = random.randint(50, 500) if tier == 'Enterprise' else random.randint(20, 100) if tier == 'Mid-Market' else random.randint(5, 30)

        # Contract terms
        contract_length_months = random.choice([12, 24, 36])
        renewal_probability = random.uniform(0.7, 0.95)

        for month in months:
            if month >= client_start_month:
                months_active = (month.year - client_start_month.year) * 12 + (month.month - client_start_month.month)

                # Calculate if renewal period
                months_to_renewal = contract_length_months - (months_active % contract_length_months)
                is_renewal_period = months_to_renewal <= 3

                # MRR with growth trend
                growth_factor = 1 + (months_active * 0.02) + random.uniform(-0.05, 0.15)
                mrr = int(initial_mrr * growth_factor)

                # Users grow with MRR
                active_users = int(initial_users * (1 + months_active * 0.03) * random.uniform(0.95, 1.05))

                # Calculate various metrics
                arr = mrr * 12

                # Customer health indicators
                nps_score = random.randint(30, 80)
                csat_score = random.uniform(3.5, 5.0)
                health_score = random.uniform(60, 95)

                # Usage metrics
                dau = int(active_users * random.uniform(0.5, 0.8))
                mau = active_users
                feature_adoption_rate = random.uniform(0.4, 0.9)
                self_service_usage = random.randint(50, 300)

                # Support metrics
                support_tickets = random.randint(5, 50)
                avg_resolution_time_hrs = random.uniform(2, 48)

                # Sales & implementation metrics
                if months_active == 0:
                    cac = random.randint(10000, 100000)
                    implementation_days = random.randint(30, 180)
                    time_to_value_days = random.randint(45, 200)
                    sales_cycle_days = random.randint(30, 180)
                else:
                    cac = 0
                    implementation_days = 0
                    time_to_value_days = 0
                    sales_cycle_days = 0

                # Professional services
                ps_hours_used = random.randint(0, 100)
                ps_hours_contracted = random.randint(40, 200)

                # Expansion/contraction
                expansion_revenue = random.randint(0, 5000) if random.random() > 0.7 else 0

                # Churn indicator
                is_churned = 1 if months_active > 6 and random.random() < 0.03 else 0

                # Products used
                products_count = random.randint(1, 5)

                # Pipeline Metrics
                open_opportunities = random.randint(0, 10) if not is_churned else 0
                pipeline_value = random.randint(10000, 200000) if open_opportunities > 0 else 0
                weighted_pipeline = pipeline_value * random.uniform(0.2, 0.8)
                win_rate = random.uniform(0.15, 0.45)
                avg_deal_cycle_days = random.randint(45, 120)

                # Coverage metrics
                quarterly_quota = mrr * 4
                pipeline_coverage = (pipeline_value / quarterly_quota) if quarterly_quota > 0 else 0

                # Renewal metrics
                renewal_date = client_start_month + pd.DateOffset(months=contract_length_months)
                next_renewal_date = renewal_date
                while next_renewal_date <= month:
                    next_renewal_date = next_renewal_date + pd.DateOffset(months=contract_length_months)

                days_to_renewal = (next_renewal_date - month).days
                renewal_risk_score = random.uniform(0, 1) if is_renewal_period else 0
                renewal_conversations_count = random.randint(0, 5) if is_renewal_period else 0

                # Channel/Partner metrics
                if acquisition_source in ['Channel Partner', 'Alliance Partner']:
                    partner_influenced_revenue = mrr * random.uniform(0.8, 1.0)
                    partner_commission_pct = random.uniform(0.10, 0.25)
                    partner_engagement_score = random.uniform(60, 95)
                else:
                    partner_influenced_revenue = 0
                    partner_commission_pct = 0
                    partner_engagement_score = 0

                # AI & Automation metrics
                ai_tools_adopted = random.randint(0, 5)
                ai_tool_usage_hours_saved = random.randint(0, 50) if ai_tools_adopted > 0 else 0
                automation_workflows_active = random.randint(0, 8)
                automated_report_views = random.randint(10, 200)

                # Self-service BI metrics
                self_service_reports_created = random.randint(5, 50)
                dashboard_views = random.randint(20, 500)
                data_quality_score = random.uniform(0.85, 1.0)
                report_generation_time_mins = random.uniform(1, 15)

                # Data model metrics
                active_data_models = random.randint(3, 15)
                data_refresh_frequency_hrs = random.choice([1, 4, 8, 24])
                data_integration_sources = random.randint(2, 10)

                # Stakeholder engagement metrics
                executive_report_requests = random.randint(1, 10)
                adhoc_analysis_requests = random.randint(2, 20)
                stakeholder_satisfaction_score = random.uniform(3.5, 5.0)
                report_delivery_sla_met_pct = random.uniform(0.85, 1.0)

                # Team productivity metrics
                reports_delivered_on_time = random.randint(15, 50)
                total_reports_requested = random.randint(20, 60)
                avg_report_turnaround_days = random.uniform(1, 7)

                # Closed deal analysis
                closed_won_deals = random.randint(0, 3) if months_active > 0 else 1
                closed_lost_deals = random.randint(0, 2)
                avg_closed_deal_size = random.randint(50000, 200000) if closed_won_deals > 0 else 0

                data.append({
                    'month': month,
                    'client_name': client,
                    'industry': industry,
                    'tier': tier,
                    'acquisition_source': acquisition_source,
                    'partner_name': partner_name,
                    'months_active': months_active,
                    'contract_length_months': contract_length_months,
                    'mrr': mrr if not is_churned else 0,
                    'arr': arr if not is_churned else 0,
                    'expansion_revenue': expansion_revenue,
                    'active_users': active_users if not is_churned else 0,
                    'dau': dau if not is_churned else 0,
                    'mau': mau if not is_churned else 0,
                    'products_count': products_count,
                    'feature_adoption_rate': feature_adoption_rate,
                    'self_service_portal_usage': self_service_usage if not is_churned else 0,
                    'support_tickets': support_tickets if not is_churned else 0,
                    'avg_resolution_time_hrs': avg_resolution_time_hrs if not is_churned else 0,
                    'nps_score': nps_score if not is_churned else 0,
                    'csat_score': csat_score if not is_churned else 0,
                    'customer_health_score': health_score if not is_churned else 0,
                    'cac': cac,
                    'sales_cycle_days': sales_cycle_days,
                    'ps_hours_contracted': ps_hours_contracted,
                    'ps_hours_used': ps_hours_used,
                    'implementation_days': implementation_days,
                    'time_to_value_days': time_to_value_days,
                    'is_churned': is_churned,
                    'open_opportunities': open_opportunities,
                    'pipeline_value': pipeline_value,
                    'weighted_pipeline': weighted_pipeline,
                    'win_rate': win_rate,
                    'avg_deal_cycle_days': avg_deal_cycle_days,
                    'pipeline_coverage': pipeline_coverage,
                    'days_to_renewal': days_to_renewal,
                    'is_renewal_period': is_renewal_period,
                    'renewal_risk_score': renewal_risk_score,
                    'renewal_conversations_count': renewal_conversations_count,
                    'renewal_probability': renewal_probability,
                    'partner_influenced_revenue': partner_influenced_revenue,
                    'partner_commission_pct': partner_commission_pct,
                    'partner_engagement_score': partner_engagement_score,
                    'ai_tools_adopted': ai_tools_adopted,
                    'ai_tool_usage_hours_saved': ai_tool_usage_hours_saved,
                    'automation_workflows_active': automation_workflows_active,
                    'automated_report_views': automated_report_views,
                    'self_service_reports_created': self_service_reports_created,
                    'dashboard_views': dashboard_views,
                    'data_quality_score': data_quality_score,
                    'report_generation_time_mins': report_generation_time_mins,
                    'active_data_models': active_data_models,
                    'data_refresh_frequency_hrs': data_refresh_frequency_hrs,
                    'data_integration_sources': data_integration_sources,
                    'executive_report_requests': executive_report_requests,
                    'adhoc_analysis_requests': adhoc_analysis_requests,
                    'stakeholder_satisfaction_score': stakeholder_satisfaction_score,
                    'report_delivery_sla_met_pct': report_delivery_sla_met_pct,
                    'reports_delivered_on_time': reports_delivered_on_time,
                    'total_reports_requested': total_reports_requested,
                    'avg_report_turnaround_days': avg_report_turnaround_days,
                    'closed_won_deals': closed_won_deals,
                    'closed_lost_deals': closed_lost_deals,
                    'avg_closed_deal_size': avg_closed_deal_size,
                    'client_start_date': client_start_month
                })

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add calculated columns
    df['ps_utilization_rate'] = (df['ps_hours_used'] / df['ps_hours_contracted']).round(2)
    df['dau_mau_ratio'] = (df['dau'] / df['mau'].replace(0, 1)).round(2)
    df['report_delivery_rate'] = (df['reports_delivered_on_time'] / df['total_reports_requested']).round(2)
    df['win_loss_ratio'] = (df['closed_won_deals'] / (df['closed_lost_deals'] + 1)).round(2)

    # Sort by client and month
    df = df.sort_values(['client_name', 'month']).reset_index(drop=True)

    return df

# ============================================================================
# STREAMLIT DASHBOARD SECTION
# ============================================================================

# Page configuration
st.set_page_config(
    page_title="Strategic Planning & Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
        <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ServiceNow Brand Colors */
    :root {
        --snow-dark-green: #293E40;
        --snow-light-green: #81B5A1;
        --snow-teal: #1F8476;
        --snow-dark-gray: #2C3E50;
        --snow-light-gray: #F4F6F8;
        --snow-success: #28B463;
        --snow-warning: #F39C12;
        --snow-error: #E74C3C;
    }

    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main container with ServiceNow gradient */
    .main {
        background: linear-gradient(135deg, #F4F6F8 0%, #E8EEF2 100%);
    }

    /* Headers with ServiceNow colors */
    h1 {
        color: var(--snow-dark-green);
        font-weight: 700;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    h2 {
        color: var(--snow-dark-green);
        font-weight: 600;
    }

    h3 {
        color: var(--snow-teal);
        font-weight: 500;
    }

    /* Glassmorphism Metric Cards with ServiceNow styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.98);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 20px 0 rgba(41, 62, 64, 0.1);
        border: 1px solid rgba(129, 181, 161, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 32px 0 rgba(41, 62, 64, 0.15);
        border-color: var(--snow-light-green);
    }

    .stMetric label {
        text-align: center;
        justify-content: center;
        display: flex;
    }

    .stMetric > div {
        text-align: center;
        justify-content: center;
    }

    [data-testid="stMetricLabel"] {
        text-align: center;
        justify-content: center;
        display: flex;
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    [data-testid="stMetricValue"] {
        text-align: center;
        justify-content: center;
        display: flex;
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--snow-teal) 0%, var(--snow-dark-green) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    [data-testid="stMetricDelta"] {
        text-align: center;
        justify-content: center;
        display: flex;
        font-weight: 600;
    }

    /* Section headers with ServiceNow brand gradient */
    .section-header {
        background: linear-gradient(135deg, var(--snow-teal) 0%, var(--snow-dark-green) 100%);
        color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0 1rem 0;
        box-shadow: 0 4px 20px 0 rgba(31, 132, 118, 0.2);
    }

    .section-header h3 {
        color: white;
        margin: 0;
        font-weight: 600;
        text-align: center;
    }

    /* Sidebar with ServiceNow branding */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, 
            var(--snow-dark-green) 0%, 
            var(--snow-teal) 50%, 
            var(--snow-light-green) 100%);
        padding: 2rem 1rem;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }

    [data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500;
    }

    /* Tabs with ServiceNow styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        color: var(--snow-dark-gray);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(129, 181, 161, 0.15);
        color: var(--snow-teal);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--snow-teal) 0%, var(--snow-dark-green) 100%) !important;
        color: white !important;
    }

    /* Buttons with ServiceNow branding */
    .stButton button {
        background: linear-gradient(135deg, var(--snow-teal) 0%, var(--snow-dark-green) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px 0 rgba(31, 132, 118, 0.25);
    }

    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px 0 rgba(31, 132, 118, 0.35);
        background: linear-gradient(135deg, var(--snow-dark-green) 0%, var(--snow-teal) 100%);
    }

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px 0 rgba(41, 62, 64, 0.1);
        background: white;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(129, 181, 161, 0.1);
        border-radius: 8px;
        font-weight: 600;
        color: var(--snow-dark-green);
        border-left: 3px solid var(--snow-light-green);
    }

    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 2px 10px 0 rgba(41, 62, 64, 0.08);
    }

    .user-message {
        background: linear-gradient(135deg, rgba(129, 181, 161, 0.15) 0%, rgba(129, 181, 161, 0.05) 100%);
        text-align: right;
        border-left: 4px solid var(--snow-light-green);
    }

    .bot-message {
        background: linear-gradient(135deg, rgba(31, 132, 118, 0.1) 0%, rgba(31, 132, 118, 0.05) 100%);
        text-align: left;
        border-left: 4px solid var(--snow-teal);
    }

    /* Charts with ServiceNow styling */
    .js-plotly-plot {
        border-radius: 12px;
        box-shadow: 0 4px 20px 0 rgba(41, 62, 64, 0.08);
        background: white;
        padding: 1rem;
        transition: box-shadow 0.3s ease;
        border: 1px solid rgba(129, 181, 161, 0.15);
    }

    .js-plotly-plot:hover {
        box-shadow: 0 8px 32px 0 rgba(31, 132, 118, 0.15);
        border-color: var(--snow-light-green);
    }

    /* Input field styling */
    .stTextInput input, .stTextArea textarea, .stSelectbox select {
        border-radius: 8px;
        border: 2px solid rgba(129, 181, 161, 0.3);
        transition: all 0.3s ease;
    }

    .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox select:focus {
        border-color: var(--snow-teal);
        box-shadow: 0 0 0 3px rgba(31, 132, 118, 0.1);
    }

    /* Scrollbar with ServiceNow colors */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: rgba(129, 181, 161, 0.1);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, var(--snow-teal) 0%, var(--snow-light-green) 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, var(--snow-dark-green) 0%, var(--snow-teal) 100%);
    }

    /* Status badges with ServiceNow colors */
    .badge-excellent {
        background: linear-gradient(135deg, var(--snow-success) 0%, #1e8449 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(40, 180, 99, 0.3);
    }

    .badge-good {
        background: linear-gradient(135deg, var(--snow-light-green) 0%, var(--snow-teal) 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(129, 181, 161, 0.3);
    }

    .badge-warning {
        background: linear-gradient(135deg, var(--snow-warning) 0%, #d68910 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(243, 156, 18, 0.3);
    }

    .badge-critical {
        background: linear-gradient(135deg, var(--snow-error) 0%, #c0392b 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 50px;
        font-size: 0.75rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(231, 76, 60, 0.3);
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .stMetric {
            padding: 1rem;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.5rem;
        }

        .section-header {
            padding: 0.5rem 1rem;
        }
    }

    /* Print styles */
    @media print {
        .stButton, [data-testid="stSidebar"], .stTabs {
            display: none;
        }

        .main {
            background: white !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load or generate data
@st.cache_data
def load_data():
    """Load data from CSV file or generate if not exists"""
    csv_file = 'llm_dashboard.csv'

    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            df['month'] = pd.to_datetime(df['month'])
            return df
        except Exception as e:
            st.warning(f"Error loading CSV: {str(e)}. Generating new data...")

    # Generate new data if file doesn't exist or has errors
    with st.spinner('Generating dataset... This may take a moment.'):
        df = generate_dataset()
        df.to_csv(csv_file, index=False)
        st.success(f'✅ Dataset generated and saved to {csv_file}')

    return df

# Load data
df = load_data()

# Title
st.title("📊 Strategic Planning & Analytics Dashboard")
st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Gemini API Key input
api_key = st.sidebar.text_input("🔑 Gemini API Key", value="AIzaSyBqxxgDhHmvjdAp-KtJk50_2OKauB47790", type="password")

min_date = df['month'].min()
max_date = df['month'].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

industries = ['All'] + sorted(df['industry'].unique().tolist())
selected_industry = st.sidebar.selectbox("Industry", industries)

tiers = ['All'] + sorted(df['tier'].unique().tolist())
selected_tier = st.sidebar.selectbox("Customer Tier", tiers)

sources = ['All'] + sorted(df['acquisition_source'].unique().tolist())
selected_source = st.sidebar.selectbox("Acquisition Source", sources)

# Apply filters
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[(filtered_df['month'] >= pd.Timestamp(date_range[0])) & 
                               (filtered_df['month'] <= pd.Timestamp(date_range[1]))]

if selected_industry != 'All':
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]

if selected_tier != 'All':
    filtered_df = filtered_df[filtered_df['tier'] == selected_tier]

if selected_source != 'All':
    filtered_df = filtered_df[filtered_df['acquisition_source'] == selected_source]

latest_month = filtered_df['month'].max()
latest_data = filtered_df[filtered_df['month'] == latest_month]
prev_month = latest_month - pd.DateOffset(months=1)
prev_data = filtered_df[filtered_df['month'] == prev_month]

# Executive Summary
st.header("📈 Executive Summary ")

# Calculate current metrics
current_mrr = latest_data['mrr'].sum()
current_pipeline = latest_data['pipeline_value'].sum()
current_customers = len(latest_data[latest_data['mrr'] > 0])
current_coverage = latest_data['pipeline_coverage'].mean()
current_users = latest_data['active_users'].sum()
current_health = latest_data[latest_data['customer_health_score'] > 0]['customer_health_score'].mean()

# Calculate previous month metrics
prev_mrr = prev_data['mrr'].sum() if len(prev_data) > 0 else 0
prev_pipeline = prev_data['pipeline_value'].sum() if len(prev_data) > 0 else 0
prev_customers = len(prev_data[prev_data['mrr'] > 0]) if len(prev_data) > 0 else 0
prev_coverage = prev_data['pipeline_coverage'].mean() if len(prev_data) > 0 else 0
prev_users = prev_data['active_users'].sum() if len(prev_data) > 0 else 0
prev_health = prev_data[prev_data['customer_health_score'] > 0]['customer_health_score'].mean() if len(prev_data) > 0 else 0

# Calculate MoM changes
mrr_change = ((current_mrr - prev_mrr) / prev_mrr * 100) if prev_mrr > 0 else 0
pipeline_change = ((current_pipeline - prev_pipeline) / prev_pipeline * 100) if prev_pipeline > 0 else 0
customers_change = current_customers - prev_customers
coverage_change = ((current_coverage - prev_coverage) / prev_coverage * 100) if prev_coverage > 0 else 0
users_change = ((current_users - prev_users) / prev_users * 100) if prev_users > 0 else 0
health_change = current_health - prev_health

# First row of metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total MRR", f"${current_mrr:,.0f}", delta=f"{mrr_change:.1f}%")

with col2:
    st.metric("Total Pipeline", f"${current_pipeline:,.0f}", delta=f"{pipeline_change:.1f}%")

with col3:
    st.metric("Active Customers", f"{current_customers}", delta=f"{customers_change:+d}")

# Second row of metrics
col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Avg Coverage", f"{current_coverage:.1f}x", delta=f"{coverage_change:.1f}%")

with col5:
    st.metric("Total Active Users", f"{current_users:,}", delta=f"{users_change:.1f}%")

with col6:
    st.metric("Avg Health Score", f"{current_health:.1f}/100", delta=f"{health_change:.1f}")

st.markdown("---")

# Tab navigation
tab1, tab2, tab3, tab4 = st.tabs([
    "💰 Revenue & Pipeline", 
    "🔄 Renewals & Churn",
    "🤝 Partners & Channels",
    "🤖 Agentic AI Assistant"
])

# TAB 1: Revenue & Pipeline
with tab1:
    st.markdown('<div class="section-header"><h3>💰 Revenue & Sales Pipeline Analysis</h3></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Calculate current values
    current_arr = latest_data['arr'].sum()
    current_pipeline_val = latest_data['pipeline_value'].sum()
    current_win_rate = latest_data['win_rate'].mean() * 100
    current_deal_size = latest_data[latest_data['avg_closed_deal_size'] > 0]['avg_closed_deal_size'].mean()

    # Calculate previous values
    prev_arr = prev_data['arr'].sum() if len(prev_data) > 0 else 0
    prev_pipeline_val = prev_data['pipeline_value'].sum() if len(prev_data) > 0 else 0
    prev_win_rate = prev_data['win_rate'].mean() * 100 if len(prev_data) > 0 else 0
    prev_deal_size = prev_data[prev_data['avg_closed_deal_size'] > 0]['avg_closed_deal_size'].mean() if len(prev_data) > 0 else 0

    # Calculate changes
    arr_change = ((current_arr - prev_arr) / prev_arr * 100) if prev_arr > 0 else 0
    pipeline_val_change = ((current_pipeline_val - prev_pipeline_val) / prev_pipeline_val * 100) if prev_pipeline_val > 0 else 0
    win_rate_change = current_win_rate - prev_win_rate
    deal_size_change = ((current_deal_size - prev_deal_size) / prev_deal_size * 100) if prev_deal_size > 0 else 0

    with col1:
        st.metric("Total ARR", f"${current_arr:,.0f}", delta=f"{arr_change:.1f}%")

    with col2:
        st.metric("Total Pipeline Value", f"${current_pipeline_val:,.0f}", delta=f"{pipeline_val_change:.1f}%")

    with col3:
        st.metric("Avg Win Rate", f"{current_win_rate:.1f}%", delta=f"{win_rate_change:.1f}pp")

    with col4:
        st.metric("Avg Deal Size", f"${current_deal_size:,.0f}", delta=f"{deal_size_change:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        # MRR Trend
        mrr_trend = filtered_df.groupby('month')['mrr'].sum().reset_index()
        fig_mrr = go.Figure()
        fig_mrr.add_trace(go.Scatter(
            x=mrr_trend['month'], y=mrr_trend['mrr'],
            mode='lines+markers', name='MRR',
            line=dict(color='#00CC96', width=3), fill='tozeroy'
        ))
        fig_mrr.update_layout(title='MRR Trend', xaxis_title='Month', yaxis_title='MRR ($)', height=400)
        st.plotly_chart(fig_mrr, use_container_width=True)

    with col2:
        # Pipeline Value Trend
        pipeline_trend = filtered_df.groupby('month')['pipeline_value'].sum().reset_index()
        fig_pipeline = go.Figure()
        fig_pipeline.add_trace(go.Scatter(
            x=pipeline_trend['month'], y=pipeline_trend['pipeline_value'],
            mode='lines+markers', name='Pipeline Value',
            line=dict(color='#636EFA', width=3), fill='tozeroy'
        ))
        fig_pipeline.update_layout(title='Pipeline Value Trend', xaxis_title='Month', yaxis_title='Pipeline ($)', height=400)
        st.plotly_chart(fig_pipeline, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        # Closed Deals Analysis
        deals_data = filtered_df.groupby('month').agg({
            'closed_won_deals': 'sum',
            'closed_lost_deals': 'sum'
        }).reset_index()
        fig_deals = go.Figure()
        fig_deals.add_trace(go.Bar(x=deals_data['month'], y=deals_data['closed_won_deals'], name='Won', marker_color='#00CC96'))
        fig_deals.add_trace(go.Bar(x=deals_data['month'], y=deals_data['closed_lost_deals'], name='Lost', marker_color='#EF553B'))
        fig_deals.update_layout(title='Closed Deals: Won vs Lost', barmode='group', height=400)
        st.plotly_chart(fig_deals, use_container_width=True)

    with col2:
        # Win Rate by Tier
        win_rate_tier = latest_data.groupby('tier')['win_rate'].mean().reset_index()
        win_rate_tier['win_rate'] = win_rate_tier['win_rate'] * 100
        fig_win_rate = px.bar(win_rate_tier, x='tier', y='win_rate', 
                               title='Win Rate by Customer Tier',
                               labels={'win_rate': 'Win Rate (%)', 'tier': 'Tier'},
                               color='win_rate', color_continuous_scale='Blues')
        fig_win_rate.update_layout(height=400)
        st.plotly_chart(fig_win_rate, use_container_width=True)

    # Pipeline Coverage Analysis
    st.subheader("📊 Pipeline Coverage Analysis")

    # Current values
    current_cov = latest_data['pipeline_coverage'].mean()
    current_weighted = latest_data['weighted_pipeline'].sum()
    current_cycle = latest_data['avg_deal_cycle_days'].mean()

    # Previous values
    prev_cov = prev_data['pipeline_coverage'].mean() if len(prev_data) > 0 else 0
    prev_weighted = prev_data['weighted_pipeline'].sum() if len(prev_data) > 0 else 0
    prev_cycle = prev_data['avg_deal_cycle_days'].mean() if len(prev_data) > 0 else 0

    # Changes
    cov_change = ((current_cov - prev_cov) / prev_cov * 100) if prev_cov > 0 else 0
    weighted_change = ((current_weighted - prev_weighted) / prev_weighted * 100) if prev_weighted > 0 else 0
    cycle_change = current_cycle - prev_cycle

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Coverage", f"{current_cov:.2f}x", delta=f"{cov_change:.1f}%", help="Pipeline Value / Quota")
    with col2:
        st.metric("Weighted Pipeline", f"${current_weighted:,.0f}", delta=f"{weighted_change:.1f}%")
    with col3:
        st.metric("Avg Deal Cycle", f"{current_cycle:.0f} days", delta=f"{cycle_change:.0f} days", delta_color="inverse")

# TAB 2: Renewals & Churn
with tab2:
    st.markdown('<div class="section-header"><h3>🔄 Renewal Management & Churn Analysis</h3></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Current values
    current_renewal_count = latest_data[latest_data['is_renewal_period'] == True].shape[0]
    current_renewal_prob = latest_data['renewal_probability'].mean() * 100
    current_high_risk = latest_data[(latest_data['is_renewal_period'] == True) & 
                                      (latest_data['renewal_risk_score'] > 0.6)].shape[0]

    prev_active = len(prev_data[prev_data['mrr'] > 0]) if len(prev_data) > 0 else 1
    current_churned = latest_data['is_churned'].sum()
    current_churn_rate = (current_churned / prev_active * 100) if prev_active > 0 else 0

    # Previous values
    prev_renewal_count = prev_data[prev_data['is_renewal_period'] == True].shape[0] if len(prev_data) > 0 else 0
    prev_renewal_prob = prev_data['renewal_probability'].mean() * 100 if len(prev_data) > 0 else 0
    prev_high_risk = prev_data[(prev_data['is_renewal_period'] == True) & 
                                (prev_data['renewal_risk_score'] > 0.6)].shape[0] if len(prev_data) > 0 else 0

    prev_prev_month = prev_month - pd.DateOffset(months=1)
    prev_prev_data = filtered_df[filtered_df['month'] == prev_prev_month]
    prev_prev_active = len(prev_prev_data[prev_prev_data['mrr'] > 0]) if len(prev_prev_data) > 0 else 1
    prev_churned = prev_data['is_churned'].sum() if len(prev_data) > 0 else 0
    prev_churn_rate = (prev_churned / prev_prev_active * 100) if prev_prev_active > 0 else 0

    # Changes
    renewal_count_change = current_renewal_count - prev_renewal_count
    renewal_prob_change = current_renewal_prob - prev_renewal_prob
    high_risk_change = current_high_risk - prev_high_risk
    churn_rate_change = current_churn_rate - prev_churn_rate

    with col1:
        st.metric("In Renewal Period", f"{current_renewal_count}", delta=f"{renewal_count_change:+d}")

    with col2:
        st.metric("Avg Renewal Probability", f"{current_renewal_prob:.1f}%", delta=f"{renewal_prob_change:.1f}pp")

    with col3:
        st.metric("High Risk Renewals", f"{current_high_risk}", delta=f"{high_risk_change:+d}", delta_color="inverse")

    with col4:
        st.metric("Churn Rate", f"{current_churn_rate:.2f}%", delta=f"{churn_rate_change:.2f}pp", delta_color="inverse")

    col1, col2 = st.columns(2)

    with col1:
        # Days to Renewal Distribution
        renewal_data = latest_data[latest_data['days_to_renewal'] < 365]
        fig_renewal = px.histogram(renewal_data, x='days_to_renewal', nbins=30,
                                    title='Days to Renewal Distribution',
                                    labels={'days_to_renewal': 'Days to Renewal'})
        fig_renewal.update_layout(height=400)
        st.plotly_chart(fig_renewal, use_container_width=True)

    with col2:
        # Renewal Risk by Tier
        risk_by_tier = latest_data[latest_data['is_renewal_period'] == True].groupby('tier')['renewal_risk_score'].mean().reset_index()
        if len(risk_by_tier) > 0:
            fig_risk = px.bar(risk_by_tier, x='tier', y='renewal_risk_score',
                              title='Average Renewal Risk Score by Tier',
                              labels={'renewal_risk_score': 'Risk Score', 'tier': 'Tier'},
                              color='renewal_risk_score', color_continuous_scale='RdYlGn_r')
            fig_risk.update_layout(height=400)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            st.info("No customers in renewal period")

    # High Risk Renewals Table
    st.subheader("⚠️ High Risk Renewals Requiring Attention")
    high_risk = latest_data[(latest_data['is_renewal_period'] == True) & 
                             (latest_data['renewal_risk_score'] > 0.5)][
        ['client_name', 'tier', 'mrr', 'days_to_renewal', 'renewal_risk_score', 
         'customer_health_score', 'renewal_conversations_count']
    ].sort_values('renewal_risk_score', ascending=False)

    if len(high_risk) > 0:
        high_risk_display = high_risk.copy()
        high_risk_display['mrr'] = high_risk_display['mrr'].apply(lambda x: f"${x:,.0f}")
        high_risk_display['renewal_risk_score'] = high_risk_display['renewal_risk_score'].apply(lambda x: f"{x:.2f}")
        high_risk_display['customer_health_score'] = high_risk_display['customer_health_score'].apply(lambda x: f"{x:.1f}")
        high_risk_display.columns = ['Client', 'Tier', 'MRR', 'Days to Renewal', 'Risk Score', 'Health Score', 'Conversations']
        st.dataframe(high_risk_display, use_container_width=True, hide_index=True)
    else:
        st.info("✅ No high-risk renewals at this time!")

# TAB 3: Partners & Channels
with tab3:
    st.markdown('<div class="section-header"><h3>🤝 Channel Partner & Alliance Performance</h3></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    # Current values
    current_partner_revenue = latest_data['partner_influenced_revenue'].sum()
    current_total_revenue = latest_data['mrr'].sum()
    current_partner_pct = (current_partner_revenue / current_total_revenue * 100) if current_total_revenue > 0 else 0
    current_partner_engagement = latest_data[latest_data['partner_engagement_score'] > 0]['partner_engagement_score'].mean()
    current_partner_count = latest_data[latest_data['partner_name'].notna()]['partner_name'].nunique()

    # Previous values
    prev_partner_revenue = prev_data['partner_influenced_revenue'].sum() if len(prev_data) > 0 else 0
    prev_total_revenue = prev_data['mrr'].sum() if len(prev_data) > 0 else 0
    prev_partner_pct = (prev_partner_revenue / prev_total_revenue * 100) if prev_total_revenue > 0 else 0
    prev_partner_engagement = prev_data[prev_data['partner_engagement_score'] > 0]['partner_engagement_score'].mean() if len(prev_data) > 0 else 0
    prev_partner_count = prev_data[prev_data['partner_name'].notna()]['partner_name'].nunique() if len(prev_data) > 0 else 0

    # Changes
    partner_revenue_change = ((current_partner_revenue - prev_partner_revenue) / prev_partner_revenue * 100) if prev_partner_revenue > 0 else 0
    partner_pct_change = current_partner_pct - prev_partner_pct
    partner_engagement_change = current_partner_engagement - prev_partner_engagement
    partner_count_change = current_partner_count - prev_partner_count

    with col1:
        st.metric("Partner Influenced Revenue", f"${current_partner_revenue:,.0f}", delta=f"{partner_revenue_change:.1f}%")

    with col2:
        st.metric("Partner Revenue %", f"{current_partner_pct:.1f}%", delta=f"{partner_pct_change:.1f}pp")

    with col3:
        st.metric("Avg Partner Engagement", f"{current_partner_engagement:.1f}/100", delta=f"{partner_engagement_change:.1f}")

    with col4:
        st.metric("Active Partners", f"{current_partner_count}", delta=f"{partner_count_change:+d}")

    col1, col2 = st.columns(2)

    with col1:
        # Revenue by Acquisition Source
        revenue_by_source = latest_data.groupby('acquisition_source')['mrr'].sum().reset_index()
        fig_source = px.pie(revenue_by_source, values='mrr', names='acquisition_source',
                            title='MRR by Acquisition Source', hole=0.4)
        fig_source.update_layout(height=400)
        st.plotly_chart(fig_source, use_container_width=True)

    with col2:
        # Partner Revenue Trend
        partner_trend = filtered_df.groupby('month')['partner_influenced_revenue'].sum().reset_index()
        fig_partner_trend = go.Figure()
        fig_partner_trend.add_trace(go.Scatter(
            x=partner_trend['month'], y=partner_trend['partner_influenced_revenue'],
            mode='lines+markers', name='Partner Revenue',
            line=dict(color='#AB63FA', width=3), fill='tozeroy'
        ))
        fig_partner_trend.update_layout(title='Partner Influenced Revenue Trend', 
                                         xaxis_title='Month', yaxis_title='Revenue ($)', height=400)
        st.plotly_chart(fig_partner_trend, use_container_width=True)

    # Top Partners Table
    st.subheader("🏆 Top Performing Partners")
    partner_performance = latest_data[latest_data['partner_name'].notna()].groupby('partner_name').agg({
        'partner_influenced_revenue': 'sum',
        'partner_engagement_score': 'mean',
        'client_name': 'count'
    }).reset_index()

    if len(partner_performance) > 0:
        partner_performance.columns = ['Partner', 'Total Revenue', 'Avg Engagement', 'Client Count']
        partner_performance = partner_performance.sort_values('Total Revenue', ascending=False).head(10)
        partner_performance['Total Revenue'] = partner_performance['Total Revenue'].apply(lambda x: f"${x:,.0f}")
        partner_performance['Avg Engagement'] = partner_performance['Avg Engagement'].apply(lambda x: f"{x:.1f}")
        st.dataframe(partner_performance, use_container_width=True, hide_index=True)
    else:
        st.info("No partner data available for the selected period")

# TAB 4: Agentic AI Assistant
with tab4:
    st.markdown('<div class="section-header"><h3>🤖 Agentic AI-Powered Data Assistant</h3></div>', unsafe_allow_html=True)


    # Initialize agentic chatbot
    if 'agentic_chatbot' not in st.session_state and api_key:
        st.session_state.agentic_chatbot = AgenticChatbot(api_key, filtered_df)

    # Initialize chat history
    if 'agentic_history' not in st.session_state:
        st.session_state.agentic_history = []

    # Chat interface
    st.subheader("💬 Chat with Agentic AI")

    # Display chat history
    for i, interaction in enumerate(st.session_state.agentic_history):
        st.markdown(f'<div class="chat-message user-message">🧑 You: {interaction["question"]}</div>', unsafe_allow_html=True)

        # Show agent workflow
        with st.expander("🔧 Agent Workflow", expanded=False):
            if interaction["results"].get("plan"):
                st.markdown("### 🎯 Orchestrator Plan")
                st.json(interaction["results"]["plan"])

            if interaction["results"].get("code"):
                st.markdown("### 💻 Generated Code")
                st.code(interaction["results"]["code"], language="python")

            if interaction["results"].get("errors"):
                st.markdown("### ⚠️ Processing Notes")
                for error in interaction["results"]["errors"]:
                    st.warning(error)

        # Show result
        if interaction["results"].get("result_df") is not None:
            st.markdown("### 📊 Data Result")
            st.dataframe(interaction["results"]["result_df"].head(20), use_container_width=True)

        # Show visualization
        if interaction["results"].get("visualization"):
            st.markdown("### 📈 Visualization")
            st.plotly_chart(interaction["results"]["visualization"], use_container_width=True)

        # Show summary
        if interaction["results"].get("summary"):
            st.markdown(f'<div class="chat-message bot-message">{interaction["results"]["summary"]}</div>', unsafe_allow_html=True)

    # Input for new question
    user_question = st.text_area("Ask a complex question:", 
                                  placeholder="Example: Show me the top 5 clients by MRR growth over the last 6 months and create a trend chart",
                                  key="agentic_input")

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_button = st.button("🚀 Execute", use_container_width=True)
    with col2:
        clear_button = st.button("🗑️ Clear Chat", use_container_width=True)

    if clear_button:
        st.session_state.agentic_history = []
        st.rerun()

    if ask_button and user_question:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar!")
        else:
            # Execute agentic workflow
            with st.spinner("🤖 Agents are working..."):
                if 'agentic_chatbot' not in st.session_state:
                    st.session_state.agentic_chatbot = AgenticChatbot(api_key, filtered_df)

                results = st.session_state.agentic_chatbot.execute(user_question)

                # Add to history
                st.session_state.agentic_history.append({
                    "question": user_question,
                    "results": results
                })

            st.rerun()

    # Quick action examples
    st.subheader("⚡ Example Questions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Revenue Analysis**
        - What is the total MRR trend?
        - Which tier has highest ARR?
        - Show revenue by industry
        """)

    with col2:
        st.markdown("""
        **Customer Insights**
        - Top 10 clients by revenue
        - Customers with low health scores
        - Churn risk analysis by tier
        """)

    with col3:
        st.markdown("""
        **Pipeline & Deals**
        - Pipeline coverage by tier
        - Win rate trends over time
        - Deal size distribution
        """)



# Footer
st.markdown("---")
st.markdown(f"**Data Range:** {filtered_df['month'].min().strftime('%B %Y')} to {filtered_df['month'].max().strftime('%B %Y')} | **Total Records:** {len(filtered_df):,} | **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
