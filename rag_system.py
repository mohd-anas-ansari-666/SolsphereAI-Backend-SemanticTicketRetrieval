import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
from typing import List, Dict, Any
import google.generativeai as genai

# Define sample tickets
SAMPLE_TICKETS = [
    {
        "id": "T1",
        "title": "Login failure on Safari for SSO users",
        "browser": "Safari 16.3",
        "os": "macOS Ventura",
        "customer_type": "Enterprise",
        "issue": "Redirect loop during SSO login",
        "resolution": "Clear cookies, update Safari settings to allow cross-site tracking."
    },
    {
        "id": "T2",
        "title": "Generic login issues",
        "browser": "All",
        "customer_type": "Mixed",
        "issue": "Password reset email not received",
        "resolution": "Whitelist support domain in email settings."
    },
    {
        "id": "T3",
        "title": "Login error specific to Chrome extensions",
        "browser": "Chrome",
        "customer_type": "SMB",
        "issue": "Conflict with password manager extension",
        "resolution": "Disable conflicting extension."
    },
    {
        "id": "T4",
        "title": "Mobile app login failure",
        "device": "iOS",
        "customer_type": "Consumer",
        "issue": "Authentication times out on slower connections",
        "resolution": "Increased timeout threshold in API and mobile client."
    },
    {
        "id": "T5",
        "title": "Enterprise dashboard access issue",
        "browser": "Firefox",
        "customer_type": "Enterprise",
        "issue": "CORS errors prevent loading of dashboard components",
        "resolution": "Updated CORS configuration on API server."
    }
]

class RAGSupportSystem:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=None):
        """
        Initialize the RAG Support System with a specified embedding model.
        
        Args:
            model_name: Name of the sentence transformer model to use
            api_key: Optional API key for LLM integration
        """
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        
        # Store for tickets and their embeddings
        self.tickets = []
        self.ticket_ids = []
        
        # Initialize LLM if API key is provided
        self.llm_available = False
        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.llm = genai.GenerativeModel('gemini-flash-1.5')
                self.llm_available = True
                print("LLM integration active with Gemini")
            except Exception as e:
                print(f"Failed to initialize LLM: {e}")
    
    def _prepare_ticket_text(self, ticket: Dict[str, Any]) -> str:
        """Create a text representation of a ticket for embedding."""
        ticket_text = f"Title: {ticket.get('title', '')}\n"
        
        # Add all other fields except 'id'
        for key, value in ticket.items():
            if key != 'id' and key != 'title':
                ticket_text += f"{key.capitalize()}: {value}\n"
                
        return ticket_text
        
    def add_tickets(self, tickets: List[Dict[str, Any]]):
        """
        Add support tickets to the system.
        
        Args:
            tickets: List of ticket dictionaries
        """
        for ticket in tickets:
            # Prepare text representation
            ticket_text = self._prepare_ticket_text(ticket)
            
            # Generate embedding
            embedding = self.model.encode(ticket_text)
            normalized_embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            self.index.add(np.array([normalized_embedding], dtype=np.float32))
            
            # Store ticket data
            self.tickets.append(ticket)
            self.ticket_ids.append(ticket['id'])
        
        print(f"Added {len(tickets)} tickets to the RAG system.")
    
    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Search for relevant tickets based on the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return
            
        Returns:
            List of relevant ticket dictionaries
        """
        # Generate query embedding
        query_embedding = self.model.encode(query)
        normalized_query = query_embedding / np.linalg.norm(query_embedding)
        
        # Search in FAISS
        k = min(top_k, len(self.tickets))
        scores, indices = self.index.search(np.array([normalized_query], dtype=np.float32), k)
        
        # Return relevant tickets
        relevant_tickets = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.tickets):
                ticket = self.tickets[idx].copy()
                ticket['relevance_score'] = float(scores[0][i])
                relevant_tickets.append(ticket)
        
        return relevant_tickets
    
    def generate_response(self, query: str, relevant_tickets: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query and relevant tickets.
        
        Args:
            query: Original user query
            relevant_tickets: List of relevant tickets
            
        Returns:
            Generated response text
        """
        if not self.llm_available:
            # Basic template-based response if no LLM
            if not relevant_tickets:
                return "No relevant tickets found for your query."
            
            response = f"Here are some suggestions based on similar issues:\n\n"
            
            for i, ticket in enumerate(relevant_tickets, 1):
                response += f"Suggestion {i}:\n"
                response += f"Based on a similar issue: {ticket['title']}\n"
                response += f"Resolution: {ticket['resolution']}\n\n"
            
            return response
        else:
            # Use LLM for more sophisticated response
            context = ""
            for i, ticket in enumerate(relevant_tickets, 1):
                context += f"Ticket {i}:\n"
                context += self._prepare_ticket_text(ticket)
                context += f"Resolution: {ticket['resolution']}\n\n"
            
            prompt = f"""
            Based on the following support tickets and the user query, provide a helpful response.
            
            User Query: {query}
            
            Relevant Support Tickets:
            {context}
            
            Please suggest possible solutions based on these similar past issues.
            """
            
            try:
                result = self.llm.generate_content(prompt)
                return result.text
            except Exception as e:
                print(f"LLM error: {e}")
                # Fall back to template
                return self.generate_response_without_llm(query, relevant_tickets)
    
    def generate_response_without_llm(self, query: str, relevant_tickets: List[Dict[str, Any]]) -> str:
        """Fallback response generator without LLM."""
        if not relevant_tickets:
            return "No relevant tickets found for your query."
        
        response = f"Here are some suggestions based on similar issues:\n\n"
        
        for i, ticket in enumerate(relevant_tickets, 1):
            response += f"Suggestion {i}:\n"
            response += f"Based on a similar issue: {ticket['title']}\n"
            if 'browser' in ticket:
                response += f"Browser: {ticket['browser']}\n"
            if 'customer_type' in ticket:
                response += f"Customer type: {ticket['customer_type']}\n"
            response += f"Resolution: {ticket['resolution']}\n\n"
        
        return response


def main():
    # Initialize the RAG system
    # To use Gemini, provide your API key: rag_system = RAGSupportSystem(api_key="YOUR_API_KEY")
    rag_system = RAGSupportSystem()
    
    # Add sample tickets
    rag_system.add_tickets(SAMPLE_TICKETS)
    
    # Interactive query loop
    print("\n===== RAG Support Ticket System =====")
    print("Enter 'exit' to quit")
    
    while True:
        query = input("\nEnter your support query: ")
        if query.lower() == 'exit':
            break
            
        # Search for relevant tickets
        relevant_tickets = rag_system.search(query, top_k=2)
        
        # Generate response
        response = rag_system.generate_response(query, relevant_tickets)
        
        # Display response
        print("\n----- Response -----")
        print(response)
        
        # Display raw relevant tickets for debugging
        print("\n----- Retrieved Tickets -----")
        for i, ticket in enumerate(relevant_tickets, 1):
            print(f"Ticket {i} (Score: {ticket['relevance_score']:.4f}):")
            print(f"ID: {ticket['id']}")
            print(f"Title: {ticket['title']}")
            print(f"Resolution: {ticket['resolution']}")
            print()


if __name__ == "__main__":
    main()