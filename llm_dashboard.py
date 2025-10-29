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
# AGENTIC AI SYSTEM
# ============================================================================

class BaseAgent:
    """Base class for all agents"""
    def __init__(self, api_key, role, instructions):
        self.api_key = api_key
        self.role = role
        self.instructions = instructions
        self.client = genai.Client(api_key=api_key) if api_key else None

    def query(self, prompt, context=""):
        """Query the Gemini API"""
        if not self.client:
            return "Error: API key not provided"

        try:
            full_prompt = f"""Role: {self.role}

Instructions: {self.instructions}

Context: {context}

Task: {prompt}
"""
            response = self.client.models.generate_content(
                model=selected_model,
                contents=full_prompt,
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class OrchestratorAgent(BaseAgent):
    """Orchestrator agent that decides which agents to call"""
    def __init__(self, api_key):
        role = "Orchestrator Agent"
        instructions = """You are an orchestrator that analyzes user questions and decides which agents to call.

Available agents:
1. python_coding_agent - For data processing, aggregations, filtering
2. plotting_agent - For creating visualizations
3. summarizing_agent - For providing executive summaries and insights

Your task:
- Analyze the user question
- Decide which agents to call and in what order
- Return a JSON with the plan

Response format:
{
    "reasoning": "Why you chose this approach",
    "agents": ["agent1", "agent2", "agent3"],
    "needs_data_processing": true/false,
    "needs_visualization": true/false,
    "needs_summary": true/false
}"""
        super().__init__(api_key, role, instructions)

    def plan_execution(self, user_question, data_summary):
        """Plan which agents to execute"""
        prompt = f"""User Question: {user_question}

Data Summary: {data_summary}

Decide which agents to call to answer this question comprehensively."""

        response = self.query(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "{" in response and "}" in response:
                json_str = response[response.find("{"):response.rfind("}")+1]
            else:
                json_str = response

            plan = json.loads(json_str)
            return plan
        except:
            # Default plan if parsing fails
            return {
                "reasoning": "Using default plan",
                "agents": ["python_coding_agent", "summarizing_agent"],
                "needs_data_processing": True,
                "needs_visualization": False,
                "needs_summary": True
            }

class PythonCodingAgent(BaseAgent):
    """Agent that writes Python code to process data"""
    def __init__(self, api_key):
        role = "Python Coding Agent"
        instructions = """You are an expert Python programmer specializing in data analysis with pandas.

Your task:
- Write clean, efficient Python code to process the dataframe
- Use pandas operations (groupby, agg, filter, etc.)
- Return ONLY executable Python code
- The input dataframe variable is named 'df'
- Store result in a variable named 'result_df'
- Do not include any explanations, only code

Example:
result_df = df.groupby('month')['mrr'].sum().reset_index()"""
        super().__init__(api_key, role, instructions)

    def generate_code(self, user_question, data_dict):
        """Generate Python code for data processing"""
        context = f"""Data Dictionary:
{json.dumps(data_dict, indent=2)}

Available columns and their descriptions are provided above."""

        prompt = f"""Generate Python pandas code to answer: {user_question}

Requirements:
- Input dataframe: df
- Output dataframe: result_df
- Use only columns available in the data dictionary
- Code should be clean and executable
- Return ONLY the code, no explanations"""

        code = self.query(prompt, context)

        # Clean code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

class PythonReviewAgent(BaseAgent):
    """Agent that reviews and fixes Python code"""
    def __init__(self, api_key):
        role = "Python Review Agent"
        instructions = """You are a code reviewer specializing in pandas and data processing.

Your task:
- Review the provided code for errors
- Fix syntax errors, logic issues, or bugs
- Ensure code follows pandas best practices
- Return the corrected code ONLY
- Do not include explanations"""
        super().__init__(api_key, role, instructions)

    def review_and_fix(self, code, error_message=""):
        """Review and fix Python code"""
        prompt = f"""Review and fix this code:

{code}

Error (if any): {error_message}

Return the corrected code only."""

        fixed_code = self.query(prompt)

        # Clean code
        if "```python" in fixed_code:
            fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
        elif "```" in fixed_code:
            fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

        return fixed_code

class PlottingAgent(BaseAgent):
    """Agent that creates visualizations"""
    def __init__(self, api_key):
        role = "Plotting Agent"
        instructions = """You are a data visualization expert using Plotly.

Your task:
- Analyze the processed dataframe
- Recommend the best chart type (line, bar, pie, scatter, etc.)
- Create Plotly code to generate the visualization
- Return code that creates a 'fig' variable

Available: plotly.express as px, plotly.graph_objects as go
Input dataframe: result_df"""
        super().__init__(api_key, role, instructions)

    def generate_plot_code(self, df_head, user_question):
        """Generate Plotly code for visualization"""
        context = f"""Dataframe preview (first 5 rows):
{df_head}

Dataframe info:
- Columns: {', '.join(df_head.columns.tolist()) if hasattr(df_head, 'columns') else 'N/A'}"""

        prompt = f"""Create a Plotly visualization to answer: {user_question}

Requirements:
- Input dataframe: result_df
- Create a figure named: fig
- Use appropriate chart type
- Add title and labels
- Return ONLY the code"""

        code = self.query(prompt, context)

        # Clean code
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0].strip()
        elif "```" in code:
            code = code.split("```")[1].split("```")[0].strip()

        return code

class SummarizingAgent(BaseAgent):
    """Agent that provides executive summaries"""
    def __init__(self, api_key):
        role = "Summarizing Agent - Executive Consultant"
        instructions = """You are a senior business consultant providing insights to executives.

Your task:
- Analyze the data results
- Provide clear, actionable insights
- Use business language (avoid technical jargon)
- Highlight key findings and recommendations
- Structure as: Key Insights, Observations, Recommendations"""
        super().__init__(api_key, role, instructions)  # FIXED: Removed 'self,' from here

    def summarize(self, user_question, data_result):
        """Generate executive summary"""
        context = f"""Data Result:
{data_result}"""

        prompt = f"""Provide an executive summary for: {user_question}

Format your response with:
## Key Insights
- Main finding 1
- Main finding 2

## Observations
- Detailed observation 1
- Detailed observation 2

## Recommendations
- Action item 1
- Action item 2"""

        summary = self.query(prompt, context)
        return summary

class AgenticChatbot:
    """Main agentic chatbot coordinator"""
    def __init__(self, api_key, df):
        self.api_key = api_key
        self.df = df

        # Initialize agents
        self.orchestrator = OrchestratorAgent(api_key)
        self.python_coder = PythonCodingAgent(api_key)
        self.python_reviewer = PythonReviewAgent(api_key)
        self.plotter = PlottingAgent(api_key)
        self.summarizer = SummarizingAgent(api_key)

    def execute(self, user_question):
        """Execute the agentic workflow"""
        results = {
            "question": user_question,
            "plan": None,
            "code": None,
            "result_df": None,
            "visualization": None,
            "summary": None,
            "errors": []
        }

        try:
            # Step 1: Orchestrator plans execution
            data_summary = f"""Dataset: {len(self.df)} records
Columns: {', '.join(self.df.columns[:10].tolist())}...
Date Range: {self.df['month'].min()} to {self.df['month'].max()}"""

            plan = self.orchestrator.plan_execution(user_question, data_summary)
            results["plan"] = plan

            # Step 2: Python Coding Agent processes data
            if plan.get("needs_data_processing", True):
                code = self.python_coder.generate_code(user_question, DATA_DICTIONARY["columns"])
                results["code"] = code

                # Try to execute code
                try:
                    local_vars = {"df": self.df, "pd": pd, "np": np}
                    exec(code, {}, local_vars)
                    result_df = local_vars.get("result_df", self.df)
                    results["result_df"] = result_df
                except Exception as e:
                    # Step 3: Python Review Agent fixes errors
                    error_msg = str(e)
                    results["errors"].append(f"Initial code error: {error_msg}")

                    fixed_code = self.python_reviewer.review_and_fix(code, error_msg)
                    results["code"] = fixed_code

                    try:
                        local_vars = {"df": self.df, "pd": pd, "np": np}
                        exec(fixed_code, {}, local_vars)
                        result_df = local_vars.get("result_df", self.df)
                        results["result_df"] = result_df
                    except Exception as e2:
                        results["errors"].append(f"Fixed code error: {str(e2)}")
                        result_df = self.df.head(10)  # Fallback
                        results["result_df"] = result_df
            else:
                result_df = self.df
                results["result_df"] = result_df

            # Step 4: Plotting Agent creates visualization
            if plan.get("needs_visualization", False) and result_df is not None:
                try:
                    plot_code = self.plotter.generate_plot_code(
                        result_df.head().to_string(), 
                        user_question
                    )

                    local_vars = {
                        "result_df": result_df, 
                        "px": px, 
                        "go": go,
                        "pd": pd
                    }
                    exec(plot_code, {}, local_vars)
                    fig = local_vars.get("fig")
                    results["visualization"] = fig
                except Exception as e:
                    results["errors"].append(f"Plotting error: {str(e)}")

            # Step 5: Summarizing Agent provides insights
            if plan.get("needs_summary", True) and result_df is not None:
                data_str = result_df.head(20).to_string()
                summary = self.summarizer.summarize(user_question, data_str)
                results["summary"] = summary

        except Exception as e:
            results["errors"].append(f"Workflow error: {str(e)}")
            results["errors"].append(traceback.format_exc())

        return results

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
    
    /* Compact metrics for single row */
    .stMetric {
        padding: 0.8rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.75rem !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
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
# Gemini Model Selection
selected_model = st.sidebar.selectbox(
    "🤖 Select Gemini Model",
    [   "gemini-2.5-pro",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ],
    index=0,
    help="Choose the Gemini model for AI assistant"
)

GEMINI_API_KEY = "AIzaSyCnu2PemH38f1iF4BdbaLcUbKbJSdxHkRE"
api_key = "AIzaSyCnu2PemH38f1iF4BdbaLcUbKbJSdxHkRE"

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
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.metric("Total MRR", f"${current_mrr:,.0f}", delta=f"{mrr_change:.1f}%")

with col2:
    st.metric("Total Pipeline", f"${current_pipeline:,.0f}", delta=f"{pipeline_change:.1f}%")

with col3:
    st.metric("Active Customers", f"{current_customers}", delta=f"{customers_change:+d}")

# Second row of metrics

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

        # # Show result
        # if interaction["results"].get("result_df") is not None:
        #     st.markdown("### 📊 Data Result")
        #     st.dataframe(interaction["results"]["result_df"].head(20), use_container_width=True)

        # Show visualization
        if interaction["results"].get("visualization"):
            st.markdown("### 📈 Visualization")
            st.plotly_chart(interaction["results"]["visualization"], use_container_width=True)

        # Show summary
        if interaction["results"].get("summary"):
            st.markdown(f'<div class="chat-message bot-message">{interaction["results"]["summary"]}</div>', unsafe_allow_html=True)

    # Input for new question
    user_question = st.text_area("Ask a question:", 
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




