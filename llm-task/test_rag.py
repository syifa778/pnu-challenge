import requests
import json
from typing import List, Dict, Any
from datetime import datetime


class RAGTester:
    """
    Comprehensive testing tool for RAG (Retrieval-Augmented Generation) system.
    Tests whether the retrieval returns appropriate results based on query type.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.retrieve_endpoint = f"{base_url}/retrieve"
        self.test_results = []
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Call the retrieve API endpoint"""
        try:
            params = {
                "query": query,
                "top_k": top_k
            }
            response = requests.post(self.retrieve_endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"✗ Error retrieving: {e}")
            return []
    
    def analyze_results(self, query: str, results: List[Dict], 
                       expected_type: str) -> Dict[str, Any]:
        """
        Analyze retrieval results to check if they match expected type.
        
        Args:
            query: The search query
            results: Retrieved results from RAG
            expected_type: Expected result type (cve, persona, cwe, etc.)
        
        Returns:
            Analysis dictionary with pass/fail status
        """
        if not results:
            return {
                "query": query,
                "expected_type": expected_type,
                "status": "FAIL",
                "reason": "No results returned",
                "retrieved_count": 0,
                "relevant_count": 0,
                "relevance_score": 0.0,
                "top_scores": [],
                "result_types": []
            }
        
        # Count relevant results based on metadata type
        relevant_count = 0
        result_types = []
        top_scores = []
        
        for result in results:
            metadata = result.get("metadata", {})
            result_type = metadata.get("type", "unknown")
            result_types.append(result_type)
            top_scores.append(result.get("score", 0.0))
            
            # Check if result type matches expected type
            if expected_type.lower() in result_type.lower():
                relevant_count += 1
            # Also check source field for CVE/CWE cases
            elif expected_type.lower() in metadata.get("source", "").lower():
                relevant_count += 1
        
        # Calculate relevance percentage
        relevance_score = (relevant_count / len(results)) * 100
        
        # Determine pass/fail (>= 60% relevant results is a pass)
        status = "PASS" if relevance_score >= 60.0 else "FAIL"
        
        # Additional checks
        reason = ""
        if relevance_score < 60.0:
            reason = f"Low relevance: only {relevant_count}/{len(results)} results match expected type '{expected_type}'"
        elif relevance_score == 100.0:
            reason = "Perfect match: all results are relevant"
        else:
            reason = f"Good match: {relevant_count}/{len(results)} results are relevant"
        
        return {
            "query": query,
            "expected_type": expected_type,
            "status": status,
            "reason": reason,
            "retrieved_count": len(results),
            "relevant_count": relevant_count,
            "relevance_score": round(relevance_score, 2),
            "top_scores": [round(s, 4) for s in top_scores[:3]],
            "result_types": result_types,
            "sample_result": {
                "text_preview": results[0].get("text", "")[:200] + "...",
                "metadata": results[0].get("metadata", {})
            } if results else None
        }
    
    def run_test_case(self, query: str, expected_type: str, 
                     top_k: int = 5, description: str = "") -> Dict[str, Any]:
        """Run a single test case"""
        print(f"\n📋 Testing: {description or query}")
        print(f"   Query: '{query}'")
        print(f"   Expected Type: {expected_type}")
        
        # Retrieve results
        results = self.retrieve(query, top_k)
        
        # Analyze results
        analysis = self.analyze_results(query, results, expected_type)
        analysis["description"] = description
        analysis["top_k"] = top_k
        
        # Print summary
        status_icon = "✓" if analysis["status"] == "PASS" else "✗"
        print(f"   {status_icon} Status: {analysis['status']}")
        print(f"   Relevance: {analysis['relevance_score']}%")
        print(f"   Result Types: {', '.join(set(analysis['result_types']))}")
        
        self.test_results.append(analysis)
        return analysis
    
    def run_cve_tests(self):
        """Test CVE-related queries"""
        print("\n" + "="*70)
        print("CVE RETRIEVAL TESTS")
        print("="*70)
        
        cve_tests = [
            {
                "query": "CVE-2025-5066",
                "expected_type": "cve",
                "description": "Specific CVE ID lookup"
            },
            {
                "query": "Chrome vulnerability 2025",
                "expected_type": "cve",
                "description": "Recent browser vulnerability"
            },
            {
                "query": "SQL injection CVE",
                "expected_type": "cve",
                "description": "SQL injection vulnerabilities"
            },
            {
                "query": "buffer overflow vulnerability",
                "expected_type": "cve",
                "description": "Buffer overflow CVEs"
            },
            {
                "query": "remote code execution",
                "expected_type": "cve",
                "description": "RCE vulnerabilities"
            },
            {
                "query": "authentication bypass CVE",
                "expected_type": "cve",
                "description": "Authentication bypass vulnerabilities"
            }
        ]
        
        for test in cve_tests:
            self.run_test_case(**test)
    
    def run_cwe_tests(self):
        """Test CWE-related queries"""
        print("\n" + "="*70)
        print("CWE RETRIEVAL TESTS")
        print("="*70)
        
        cwe_tests = [
            {
                "query": "CWE-451",
                "expected_type": "cwe",
                "description": "Specific CWE ID lookup"
            },
            {
                "query": "improper authentication weakness",
                "expected_type": "cwe",
                "description": "Authentication weakness pattern"
            },
            {
                "query": "cross-site scripting weakness",
                "expected_type": "cwe",
                "description": "XSS weakness pattern"
            },
            {
                "query": "CWE-79 XSS",
                "expected_type": "cwe",
                "description": "Specific XSS CWE"
            }
        ]
        
        for test in cwe_tests:
            self.run_test_case(**test)
    
    def run_persona_tests(self):
        """Test personal information retrieval"""
        print("\n" + "="*70)
        print("PERSONA/PERSONAL INFORMATION RETRIEVAL TESTS")
        print("="*70)
        
        persona_tests = [
            {
                "query": "Alicia Mosquera",
                "expected_type": "persona",
                "description": "Person by full name"
            },
            {
                "query": "person in Coalinga California",
                "expected_type": "persona",
                "description": "Person by location"
            },
            {
                "query": "William Wood",
                "expected_type": "persona",
                "description": "Another person by name"
            },
            {
                "query": "Sean Buckley Irish",
                "expected_type": "persona",
                "description": "Person by name and nationality"
            },
            {
                "query": "Kim Boissiere",
                "expected_type": "persona",
                "description": "Person associated with email"
            },
            {
                "query": "retired person who gardens",
                "expected_type": "persona",
                "description": "Person by characteristics"
            },
            {
                "query": "female 78 years old California",
                "expected_type": "persona",
                "description": "Person by demographics"
            }
        ]
        
        for test in persona_tests:
            self.run_test_case(**test)
    
    def run_mixed_tests(self):
        """Test queries that could return mixed results"""
        print("\n" + "="*70)
        print("MIXED/EDGE CASE TESTS")
        print("="*70)
        
        mixed_tests = [
            {
                "query": "security vulnerability in FTP server",
                "expected_type": "cve",
                "description": "Generic security query (should return CVE)"
            },
            {
                "query": "expert in cybersecurity",
                "expected_type": "persona",
                "description": "Expertise query (should return persona)"
            },
            {
                "query": "D-Link device vulnerability",
                "expected_type": "cve",
                "description": "Vendor-specific vulnerability"
            },
            {
                "query": "Filmora 14.5.16 security issue",
                "expected_type": "cve",
                "description": "Software-specific security"
            }
        ]
        
        for test in mixed_tests:
            self.run_test_case(**test)
    
    def run_all_tests(self, top_k: int = 5) -> Dict[str, Any]:
        """Run all test suites"""
        print("="*70)
        print("RAG RETRIEVAL TESTING - COMPREHENSIVE SUITE")
        print("="*70)
        print(f"Testing with top_k={top_k}")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Run all test categories
        self.run_cve_tests()
        self.run_cwe_tests()
        self.run_persona_tests()
        self.run_mixed_tests()
        
        # Calculate summary statistics
        summary = self.calculate_summary()
        
        return summary
    
    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics from all test results"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["status"] == "PASS")
        failed_tests = total_tests - passed_tests
        
        # Calculate average relevance score
        avg_relevance = sum(r["relevance_score"] for r in self.test_results) / total_tests if total_tests > 0 else 0
        
        # Group by expected type
        by_type = {}
        for result in self.test_results:
            exp_type = result["expected_type"]
            if exp_type not in by_type:
                by_type[exp_type] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_relevance": 0
                }
            by_type[exp_type]["total"] += 1
            if result["status"] == "PASS":
                by_type[exp_type]["passed"] += 1
            else:
                by_type[exp_type]["failed"] += 1
        
        # Calculate average relevance by type
        for exp_type in by_type:
            type_results = [r for r in self.test_results if r["expected_type"] == exp_type]
            by_type[exp_type]["avg_relevance"] = round(
                sum(r["relevance_score"] for r in type_results) / len(type_results), 2
            )
        
        # Identify failed tests
        failed_test_details = [
            {
                "description": r["description"],
                "query": r["query"],
                "expected_type": r["expected_type"],
                "relevance_score": r["relevance_score"],
                "reason": r["reason"]
            }
            for r in self.test_results if r["status"] == "FAIL"
        ]
        
        summary = {
            "test_metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "pass_rate": round((passed_tests / total_tests * 100), 2) if total_tests > 0 else 0,
                "average_relevance_score": round(avg_relevance, 2)
            },
            "results_by_type": by_type,
            "failed_tests": failed_test_details,
            "detailed_results": self.test_results
        }
        
        return summary
    
    def save_results_to_json(self, summary: Dict[str, Any], 
                            filename: str = None) -> str:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_test_results_{timestamp}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as jsonfile:
                json.dump(summary, jsonfile, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {filename}")
            return filename
        except Exception as e:
            print(f"\n✗ Error saving JSON: {e}")
            return None
    
    def print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of test results"""
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        metadata = summary["test_metadata"]
        print(f"\nTimestamp: {metadata['timestamp']}")
        print(f"Total Tests: {metadata['total_tests']}")
        print(f"Passed: {metadata['passed_tests']} ✓")
        print(f"Failed: {metadata['failed_tests']} ✗")
        print(f"Pass Rate: {metadata['pass_rate']}%")
        print(f"Average Relevance Score: {metadata['average_relevance_score']}%")
        
        print("\n" + "-"*70)
        print("RESULTS BY TYPE")
        print("-"*70)
        
        for exp_type, stats in summary["results_by_type"].items():
            print(f"\n{exp_type.upper()}:")
            print(f"  Total: {stats['total']}")
            print(f"  Passed: {stats['passed']} ✓")
            print(f"  Failed: {stats['failed']} ✗")
            print(f"  Avg Relevance: {stats['avg_relevance']}%")
        
        if summary["failed_tests"]:
            print("\n" + "-"*70)
            print("FAILED TESTS DETAILS")
            print("-"*70)
            for i, failed in enumerate(summary["failed_tests"], 1):
                print(f"\n{i}. {failed['description']}")
                print(f"   Query: '{failed['query']}'")
                print(f"   Expected: {failed['expected_type']}")
                print(f"   Relevance: {failed['relevance_score']}%")
                print(f"   Reason: {failed['reason']}")


def main():
    """Main execution function"""
    print("="*70)
    print("RAG RETRIEVAL QUALITY TESTING")
    print("="*70)
    print("\nThis test suite evaluates whether the RAG system returns")
    print("appropriate results based on query type:")
    print("  • CVE queries should return CVE results")
    print("  • CWE queries should return CWE results")
    print("  • Personal info queries should return persona results")
    print("="*70)
    
    # Initialize tester
    tester = RAGTester(base_url="http://127.0.0.1:8000")
    
    # Run all tests
    summary = tester.run_all_tests(top_k=5)
    
    # Print summary
    tester.print_summary(summary)
    
    # Save to JSON
    json_file = tester.save_results_to_json(summary)
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {json_file}")
    print(f"Pass Rate: {summary['test_metadata']['pass_rate']}%")
    
    # Return summary for programmatic access
    return summary


if __name__ == "__main__":
    main()