import streamlit as st
from rag_system import RAGSupportSystem, SAMPLE_TICKETS
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env file
# Set page title and configuration
st.set_page_config(
    page_title="RAG Support Ticket System",
    layout="wide"
)

# Initialize session state variables
if 'rag_system' not in st.session_state:
    # Check if API key is provided
    api_key = os.getenv("GEMINI_API_KEY")
    
    # Initialize the RAG system
    st.session_state.rag_system = RAGSupportSystem(api_key=api_key)
    st.session_state.rag_system.add_tickets(SAMPLE_TICKETS)

# App header
st.title("RAG-Based Support Ticket System")
st.markdown("""
This system uses RAG (Retrieval-Augmented Generation) to find relevant support tickets 
and provide contextual suggestions for customer issues.
""")

# Input section
query = st.text_input("Enter your support query:", 
                      placeholder="e.g., Login error on Safari browser for enterprise users")
top_k = st.slider("Number of tickets to retrieve:", min_value=1, max_value=5, value=2)

# Process query when submitted
if query:
    with st.spinner("Searching for relevant tickets..."):
        # Retrieve relevant tickets
        relevant_tickets = st.session_state.rag_system.search(query, top_k=top_k)
        
        # Generate response
        response = st.session_state.rag_system.generate_response(query, relevant_tickets)
        
    # Display response
    st.subheader("Generated Response")
    st.write(response)
    
    # Display retrieved tickets with relevance scores
    st.subheader("Retrieved Support Tickets")
    
    if not relevant_tickets:
        st.info("No relevant tickets found.")
    else:
        for i, ticket in enumerate(relevant_tickets):
            with st.expander(f"Ticket {i+1}: {ticket['title']} (Relevance: {ticket['relevance_score']:.4f})"):
                # Create two columns
                col1, col2 = st.columns(2)
                
                # Display ticket details in left column
                with col1:
                    st.markdown("**Ticket Details:**")
                    for key, value in ticket.items():
                        if key not in ['relevance_score', 'id', 'resolution']:
                            st.markdown(f"**{key.capitalize()}:** {value}")
                
                # Display resolution in right column
                with col2:
                    st.markdown("**Resolution:**")
                    st.markdown(f"{ticket['resolution']}")
                
                # Add feedback buttons (non-functional in this prototype)
                col_helpful, col_not_helpful = st.columns(2)
                with col_helpful:
                    st.button("👍 Helpful", key=f"helpful_{i}", disabled=True)
                with col_not_helpful:
                    st.button("👎 Not Helpful", key=f"not_helpful_{i}", disabled=True)

# Show system information in the sidebar
with st.sidebar:
    st.header("System Information")
    st.markdown("**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2")
    st.markdown("**Vector Store:** FAISS (Facebook AI Similarity Search)")
    st.markdown("**Retrieval Strategy:** Cosine similarity with top-k selection")
    
    # LLM information
    if st.session_state.rag_system.llm_available:
        st.markdown("**LLM:** Google Generative AI (Gemini-flash-1.5)")
    else:
        st.markdown("**LLM:** Not configured (using template-based responses)")
    
    # Sample tickets information
    st.subheader("Available Sample Tickets")
    for ticket in SAMPLE_TICKETS:
        with st.expander(f"{ticket['id']}: {ticket['title']}"):
            for key, value in ticket.items():
                if key not in ['id']:
                    st.markdown(f"**{key.capitalize()}:** {value}")