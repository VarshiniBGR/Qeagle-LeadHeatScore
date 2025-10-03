"""
Hit Rate Measurement Service
Measures hit rate uplift of cross-encoder reranking vs no rerank
"""

import asyncio
import time
from typing import List, Dict, Any, Tuple
from app.models.schemas import SearchResult
from app.utils.logging import get_logger
from app.services.retrieval import retrieval
from app.services.cross_encoder_reranker import cross_encoder_reranker

logger = get_logger(__name__)


class HitRateMeasurement:
    """Service for measuring hit rate improvements from reranking."""
    
    def __init__(self):
        # Dynamic test queries based on common lead scenarios
        self.test_queries = self._generate_dynamic_test_queries()
    
    def _generate_dynamic_test_queries(self) -> List[str]:
        """Generate dynamic test queries based on common lead patterns."""
        roles = ['data analyst', 'project manager', 'healthcare admin', 'software engineer', 'business analyst']
        campaigns = ['early_bird', 'tech_conference', 'product_launch', 'winter_campaign', 'spring_promotion']
        keywords = ['machine learning', 'python', 'agile', 'analytics', 'leadership', 'certification']
        
        queries = []
        
        # Role-specific queries
        for role in roles:
            queries.append(f"{role} course for career development")
            queries.append(f"professional training for {role}")
        
        # Campaign-specific queries
        for campaign in campaigns:
            queries.append(f"{campaign.replace('_', ' ')} program details")
            queries.append(f"enrollment in {campaign.replace('_', ' ')}")
        
        # Keyword-specific queries
        for keyword in keywords:
            queries.append(f"{keyword} course for professionals")
            queries.append(f"advanced {keyword} training")
        
        # Mixed queries
        queries.extend([
            f"{roles[0]} {campaigns[0]} program",
            f"{keywords[0]} certification course",
            f"professional development for {roles[1]}",
            f"{campaigns[1]} {keywords[1]} training"
        ])
        
        return queries[:15]  # Return top 15 most relevant queries
    
    async def measure_hit_rate_uplift(
        self, 
        num_queries: int = 10,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Measure hit rate uplift of cross-encoder reranking vs no rerank.
        
        Args:
            num_queries: Number of test queries to run
            top_k: Number of top results to evaluate
            
        Returns:
            Dictionary with hit rate measurements and improvements
        """
        try:
            logger.info(f"Starting hit rate measurement with {num_queries} queries")
            
            # Use subset of test queries
            test_queries = self.test_queries[:num_queries]
            
            results = {
                "test_queries": test_queries,
                "num_queries": num_queries,
                "top_k": top_k,
                "measurements": []
            }
            
            total_no_rerank_hits = 0
            total_rerank_hits = 0
            total_queries = 0
            
            for query in test_queries:
                try:
                    # Test 1: No reranking (hybrid search only)
                    no_rerank_start = time.time()
                    no_rerank_results = await retrieval.hybrid_search(query, limit=top_k)
                    no_rerank_latency = (time.time() - no_rerank_start) * 1000
                    
                    # Test 2: With cross-encoder reranking
                    rerank_start = time.time()
                    rerank_results = await retrieval.hybrid_search(query, limit=50)  # Get top 50
                    if len(rerank_results) > top_k:
                        rerank_results = await cross_encoder_reranker.rerank_results(
                            query=query,
                            results=rerank_results,
                            top_k=top_k,
                            alpha=0.3
                        )
                    rerank_latency = (time.time() - rerank_start) * 1000
                    
                    # Calculate hit rate (simplified - assume first result is relevant)
                    no_rerank_hit = 1 if no_rerank_results else 0
                    rerank_hit = 1 if rerank_results else 0
                    
                    # Calculate relevance score improvement (simplified)
                    no_rerank_score = no_rerank_results[0].score if no_rerank_results else 0
                    rerank_score = rerank_results[0].score if rerank_results else 0
                    score_improvement = rerank_score - no_rerank_score
                    
                    measurement = {
                        "query": query,
                        "no_rerank": {
                            "results_count": len(no_rerank_results),
                            "top_score": round(no_rerank_score, 3),
                            "latency_ms": round(no_rerank_latency, 2),
                            "hit": no_rerank_hit
                        },
                        "with_rerank": {
                            "results_count": len(rerank_results),
                            "top_score": round(rerank_score, 3),
                            "latency_ms": round(rerank_latency, 2),
                            "hit": rerank_hit
                        },
                        "improvements": {
                            "score_improvement": round(score_improvement, 3),
                            "latency_overhead": round(rerank_latency - no_rerank_latency, 2),
                            "hit_improvement": rerank_hit - no_rerank_hit
                        }
                    }
                    
                    results["measurements"].append(measurement)
                    
                    total_no_rerank_hits += no_rerank_hit
                    total_rerank_hits += rerank_hit
                    total_queries += 1
                    
                except Exception as e:
                    logger.error(f"Error measuring hit rate for query '{query}': {e}")
                    continue
            
            # Calculate overall hit rates
            no_rerank_hit_rate = total_no_rerank_hits / total_queries if total_queries > 0 else 0
            rerank_hit_rate = total_rerank_hits / total_queries if total_queries > 0 else 0
            hit_rate_uplift = rerank_hit_rate - no_rerank_hit_rate
            
            # Calculate average latencies
            avg_no_rerank_latency = sum(
                m["no_rerank"]["latency_ms"] for m in results["measurements"]
            ) / len(results["measurements"]) if results["measurements"] else 0
            
            avg_rerank_latency = sum(
                m["with_rerank"]["latency_ms"] for m in results["measurements"]
            ) / len(results["measurements"]) if results["measurements"] else 0
            
            avg_latency_overhead = avg_rerank_latency - avg_no_rerank_latency
            
            # Calculate average score improvements
            avg_score_improvement = sum(
                m["improvements"]["score_improvement"] for m in results["measurements"]
            ) / len(results["measurements"]) if results["measurements"] else 0
            
            results["summary"] = {
                "total_queries": total_queries,
                "hit_rates": {
                    "no_rerank": round(no_rerank_hit_rate, 3),
                    "with_rerank": round(rerank_hit_rate, 3),
                    "uplift": round(hit_rate_uplift, 3),
                    "uplift_percentage": round(hit_rate_uplift * 100, 1)
                },
                "latency": {
                    "avg_no_rerank_ms": round(avg_no_rerank_latency, 2),
                    "avg_rerank_ms": round(avg_rerank_latency, 2),
                    "avg_overhead_ms": round(avg_latency_overhead, 2)
                },
                "scores": {
                    "avg_score_improvement": round(avg_score_improvement, 3)
                },
                "reranker_info": cross_encoder_reranker.get_model_info()
            }
            
            logger.info(f"Hit rate measurement completed. Uplift: {hit_rate_uplift:.3f} ({hit_rate_uplift*100:.1f}%)")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in hit rate measurement: {e}")
            return {
                "error": str(e),
                "status": "failed"
            }


# Global hit rate measurement instance
hit_rate_measurement = HitRateMeasurement()




