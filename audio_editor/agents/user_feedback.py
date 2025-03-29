"""
User Feedback module for the audio processing multi-agent system.
This module provides utilities for requesting and handling user feedback.
"""
import logfire
import time
import sys
import os
from typing import List, Optional, Dict, Any, Callable, Union

from .models import UserFeedbackRequest, UserFeedbackResponse


class FeedbackManager:
    """Manages requests for user feedback and their responses."""
    
    def __init__(self, interactive: bool = True):
        """Initialize the feedback manager.
        
        Args:
            interactive: Whether to enable interactive feedback (if False, uses default responses)
        """
        self.interactive = interactive
        self.default_responses = {
            "clarification": "Please proceed with your best judgment.",
            "confirmation": "yes",
            "choice": "0"  # First option by default
        }
        
    def request_feedback(self, request: UserFeedbackRequest) -> UserFeedbackResponse:
        """Request feedback from the user.
        
        Args:
            request: The feedback request
            
        Returns:
            The user's response
        """
        with logfire.span("request_user_feedback", request_type=request.request_type):
            if not self.interactive:
                # Use default responses in non-interactive mode
                response = self.default_responses.get(
                    request.request_type, 
                    "Please proceed with your best judgment."
                )
                return UserFeedbackResponse(
                    response=response,
                    timestamp=time.time()
                )
                
            # Format the request based on severity
            severity_prefix = {
                "info": "\n[INFO] ",
                "warning": "\n[WARNING] ",
                "error": "\n[ERROR] "
            }.get(request.severity, "\n[INFO] ")
            
            # Print the request to the user
            print(f"{severity_prefix}{request.query}")
            print(f"\nContext: {request.context}")
            
            # Show options if provided
            if request.options:
                print("\nOptions:")
                for i, option in enumerate(request.options):
                    print(f"  [{i}] {option}")
                print("\nEnter the number of your choice, or provide a different response:")
            else:
                print("\nPlease provide your response:")
                
            # Get the user's response
            response = input("> ").strip()
            
            return UserFeedbackResponse(
                response=response,
                timestamp=time.time()
            )
            
    def request_clarification(
        self, 
        query: str, 
        context: str, 
        severity: str = "info"
    ) -> UserFeedbackResponse:
        """Request clarification from the user.
        
        Args:
            query: The question to ask
            context: Context for the question
            severity: Severity level (info, warning, error)
            
        Returns:
            The user's response
        """
        request = UserFeedbackRequest(
            query=query,
            context=context,
            severity=severity,
            request_type="clarification"
        )
        return self.request_feedback(request)
        
    def request_confirmation(
        self, 
        query: str, 
        context: str, 
        severity: str = "info"
    ) -> bool:
        """Request confirmation from the user (yes/no question).
        
        Args:
            query: The question to ask
            context: Context for the question
            severity: Severity level (info, warning, error)
            
        Returns:
            Boolean indicating user's confirmation (True for yes, False for no)
        """
        request = UserFeedbackRequest(
            query=f"{query} (yes/no)",
            context=context,
            options=["Yes", "No"],
            severity=severity,
            request_type="confirmation"
        )
        
        response = self.request_feedback(request)
        
        # Check for numeric response referring to options
        if response.response.isdigit():
            option_index = int(response.response)
            if option_index < 2:  # We have 2 options: Yes or No
                return option_index == 0  # 0 means Yes
                
        # Check for yes/no text response
        if response.response.lower() in ["yes", "y", "true"]:
            return True
        elif response.response.lower() in ["no", "n", "false"]:
            return False
            
        # Default to True for other responses
        return True
        
    def request_choice(
        self, 
        query: str, 
        context: str, 
        options: List[str], 
        severity: str = "info"
    ) -> str:
        """Request a choice from a list of options.
        
        Args:
            query: The question to ask
            context: Context for the question
            options: List of options to choose from
            severity: Severity level (info, warning, error)
            
        Returns:
            The selected option (or user-provided response)
        """
        request = UserFeedbackRequest(
            query=query,
            context=context,
            options=options,
            severity=severity,
            request_type="choice"
        )
        
        response = self.request_feedback(request)
        
        # Check if response is a valid option index
        if response.response.isdigit():
            option_index = int(response.response)
            if 0 <= option_index < len(options):
                return options[option_index]
                
        # Otherwise, return the raw response
        return response.response


class ConsoleUserFeedbackHandler:
    """Console-based implementation of the user feedback handler."""
    
    def __init__(self, interactive: bool = True):
        """Initialize the console feedback handler.
        
        Args:
            interactive: Whether to enable interactive feedback
        """
        self.feedback_manager = FeedbackManager(interactive=interactive)
        
    def __call__(self, request: UserFeedbackRequest) -> UserFeedbackResponse:
        """Handle a feedback request by getting a response from the user.
        
        Args:
            request: The feedback request
            
        Returns:
            The user's response
        """
        return self.feedback_manager.request_feedback(request) 