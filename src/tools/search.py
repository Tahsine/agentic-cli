import os
from tavily import TavilyClient
from typing import List, Dict, Any

class TavilySearch:
    """
    Wrapper around Tavily Client for developer-centric web search.
    Requires TAVILY_API_KEY in environment variables.
    """
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
             # Basic check, though it might be set later or we might want to warn
             pass
        self.client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> str:
        """
        Perform a search and return formatted results.
        """
        try:
            # search_depth="advanced" gives better results for dev topics
            response = self.client.search(query=query, search_depth="advanced")
            results = response.get("results", [])
            
            formatted = []
            for res in results:
                title = res.get('title', 'No Title')
                content = res.get('content', '')
                url = res.get('url', '')
                formatted.append(f"Title: {title}\nSource: {url}\nContent: {content}\n")
            
            if not formatted:
                return "No results found."
                
            return "\n---\n".join(formatted)
        except Exception as e:
            return f"Search failed: {str(e)}. (Check TAVILY_API_KEY?)"
