# Fuzzy Matching Pipeline Components Architecture

## Design Status: ARCHITECTURAL DESIGN PHASE
**Task**: 32.7 - Fuzzy Matching Pipeline Components Design  
**Context**: Building on established architectural framework with intention-driven interface and streaming-first design

## Executive Summary

Fuzzy matching represents one of LocalData MCP's strongest domains with exceptional library coverage spanning string similarity, semantic matching, and entity resolution. This architecture delivers a comprehensive pipeline system that transforms complex similarity analysis from low-level algorithmic operations into high-level analytical intentions, enabling LLM agents to express needs like "find duplicate customers with 85% confidence" rather than managing Levenshtein distances and blocking strategies.

## Sequential Architecture Design

### Component 1: String Similarity Pipeline Blocks

**Purpose**: Progressive complexity string matching from simple character similarity to advanced phonetic matching

**Core Architecture**:
```python
class StringSimilarityPipeline:
    """
    Comprehensive string similarity pipeline supporting multiple algorithms with progressive complexity.
    
    Intention-Driven Interface:
    - "Find similar customer names" → automatic algorithm selection + threshold tuning
    - "Detect typos in product descriptions" → edit distance algorithms
    - "Match names across different systems" → phonetic + semantic combination
    
    Streaming Architecture:
    - Processes large string datasets in configurable chunks
    - Memory-efficient comparison matrices with intelligent blocking
    - Real-time similarity scoring for interactive applications
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.algorithm_registry = self._initialize_algorithms()
        self.blocking_strategies = BlockingStrategyManager()
        
    def create_similarity_composition(
        self, 
        intent: str,
        data_context: Dict,
        performance_requirements: Optional[Dict] = None
    ) -> SimilarityComposition:
        """
        Create string similarity pipeline based on analytical intent.
        
        Intent Examples:
        - "exact_match_with_typos": Levenshtein + Jaro-Winkler
        - "fuzzy_name_matching": Soundex + Double Metaphone + N-gram
        - "product_deduplication": Jaccard + Cosine + manual review queue
        - "address_standardization": Edit distance + phonetic + geospatial validation
        """
        parsed_intent = self._parse_similarity_intent(intent)
        
        composition = SimilarityComposition()
        
        # Algorithm Selection Strategy
        if parsed_intent.category == 'name_matching':
            composition.add_stage(
                algorithm='jaro_winkler',
                weight=0.4,
                threshold=0.85,
                rationale="Primary algorithm for name similarity"
            )
            composition.add_stage(
                algorithm='double_metaphone',
                weight=0.3,
                threshold=0.8,
                rationale="Phonetic similarity for name variations"
            )
            composition.add_stage(
                algorithm='n_gram_similarity',
                weight=0.3,
                n=2,
                threshold=0.7,
                rationale="Character sequence similarity"
            )
            
        elif parsed_intent.category == 'product_matching':
            composition.add_stage(
                algorithm='jaccard_similarity',
                weight=0.35,
                threshold=0.6,
                rationale="Token-based similarity for product names"
            )
            composition.add_stage(
                algorithm='cosine_similarity',
                weight=0.35,
                threshold=0.7,
                rationale="TF-IDF based semantic similarity"
            )
            composition.add_stage(
                algorithm='levenshtein_ratio',
                weight=0.3,
                threshold=0.8,
                rationale="Edit distance for exact variations"
            )
            
        # Performance Optimization
        if data_context.get('size', 0) > 100000:
            composition.enable_blocking_strategy('sorted_neighborhood')
            composition.set_chunk_size(10000)
            
        return composition

class StringAlgorithmRegistry:
    """
    Registry of string similarity algorithms with unified interface and performance characteristics.
    """
    
    def __init__(self):
        self.algorithms = {
            # Edit Distance Family
            'levenshtein_distance': {
                'implementation': self._levenshtein_impl,
                'complexity': 'O(n*m)',
                'best_for': ['typo_detection', 'exact_variations'],
                'libraries': ['python-Levenshtein', 'textdistance'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'damerau_levenshtein': {
                'implementation': self._damerau_levenshtein_impl,
                'complexity': 'O(n*m)',
                'best_for': ['transposition_errors', 'keyboard_mistakes'],
                'libraries': ['textdistance', 'jellyfish'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            
            # Jaro Family
            'jaro_similarity': {
                'implementation': self._jaro_impl,
                'complexity': 'O(n*m)',
                'best_for': ['name_matching', 'short_strings'],
                'libraries': ['jellyfish', 'textdistance'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'jaro_winkler': {
                'implementation': self._jaro_winkler_impl,
                'complexity': 'O(n*m)',
                'best_for': ['name_matching', 'prefix_similarity'],
                'libraries': ['jellyfish', 'fuzzywuzzy'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            
            # Token-Based Similarity
            'jaccard_similarity': {
                'implementation': self._jaccard_impl,
                'complexity': 'O(n+m)',
                'best_for': ['set_comparison', 'product_names'],
                'libraries': ['textdistance', 'sklearn'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'dice_coefficient': {
                'implementation': self._dice_impl,
                'complexity': 'O(n+m)',
                'best_for': ['bigram_similarity', 'document_similarity'],
                'libraries': ['textdistance'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            
            # Phonetic Algorithms
            'soundex': {
                'implementation': self._soundex_impl,
                'complexity': 'O(n)',
                'best_for': ['english_names', 'basic_phonetic'],
                'libraries': ['jellyfish', 'fuzzy'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'metaphone': {
                'implementation': self._metaphone_impl,
                'complexity': 'O(n)',
                'best_for': ['english_pronunciation', 'name_variants'],
                'libraries': ['jellyfish', 'fuzzy'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'double_metaphone': {
                'implementation': self._double_metaphone_impl,
                'complexity': 'O(n)',
                'best_for': ['multilingual_phonetic', 'advanced_names'],
                'libraries': ['jellyfish', 'fuzzy'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            'nysiis': {
                'implementation': self._nysiis_impl,
                'complexity': 'O(n)',
                'best_for': ['new_york_phonetic', 'census_matching'],
                'libraries': ['jellyfish'],
                'streaming_compatible': True,
                'parallel_safe': True
            },
            
            # Sequence-Based
            'n_gram_similarity': {
                'implementation': self._ngram_impl,
                'complexity': 'O(n+m)',
                'best_for': ['character_sequences', 'partial_matches'],
                'libraries': ['textdistance', 'nltk'],
                'streaming_compatible': True,
                'parallel_safe': True,
                'parameters': ['n']
            }
        }

    def _levenshtein_impl(self, s1: str, s2: str, **kwargs) -> float:
        """Optimized Levenshtein implementation with library fallback."""
        try:
            import Levenshtein
            distance = Levenshtein.distance(s1, s2)
            max_len = max(len(s1), len(s2))
            return 1.0 - (distance / max_len) if max_len > 0 else 1.0
        except ImportError:
            # Fallback to textdistance
            import textdistance
            return textdistance.levenshtein.normalized_similarity(s1, s2)
```

**Performance Optimization Engine**:
```python
class SimilarityPerformanceOptimizer:
    """
    Optimizes string similarity computations for large-scale processing.
    
    Key Optimizations:
    1. Intelligent blocking to reduce O(n²) comparisons
    2. Early termination for low-similarity pairs
    3. Parallel processing with shared memory optimization
    4. Caching of expensive phonetic transformations
    """
    
    def __init__(self):
        self.blocking_strategies = {
            'sorted_neighborhood': SortedNeighborhoodBlocking(),
            'canopy_clustering': CanopyClusteringBlocking(),
            'locality_sensitive_hashing': LSHBlocking(),
            'first_character': FirstCharacterBlocking()
        }
        
    def optimize_similarity_pipeline(
        self, 
        data_size: int,
        algorithm_composition: SimilarityComposition,
        memory_constraints: Dict
    ) -> OptimizedSimilarityPipeline:
        """
        Apply performance optimizations based on data size and constraints.
        """
        optimized_pipeline = OptimizedSimilarityPipeline()
        
        # Blocking Strategy Selection
        if data_size > 1000000:  # > 1M records
            # Use multiple blocking strategies for maximum recall
            optimized_pipeline.add_blocking_strategy('canopy_clustering', weight=0.6)
            optimized_pipeline.add_blocking_strategy('sorted_neighborhood', weight=0.4)
            
        elif data_size > 100000:  # > 100K records  
            optimized_pipeline.add_blocking_strategy('sorted_neighborhood')
            
        # Parallel Processing Configuration
        if data_size > 50000:
            optimized_pipeline.enable_parallel_processing(
                max_workers=min(8, data_size // 10000),
                chunk_size=max(1000, data_size // 100)
            )
            
        # Early Termination Thresholds
        min_threshold = min(stage.threshold for stage in algorithm_composition.stages)
        optimized_pipeline.set_early_termination_threshold(min_threshold * 0.5)
        
        # Caching Strategy
        if algorithm_composition.has_phonetic_algorithms():
            optimized_pipeline.enable_phonetic_caching(
                cache_size=min(100000, data_size // 10)
            )
            
        return optimized_pipeline

class BlockingStrategyManager:
    """
    Manages blocking strategies to reduce computational complexity of similarity comparisons.
    """
    
    def create_sorted_neighborhood_blocks(
        self, 
        data: List[str],
        window_size: int = 20,
        sort_key: Callable = None
    ) -> List[Block]:
        """
        Create blocks using sorted neighborhood algorithm.
        
        Algorithm:
        1. Sort strings by specified key (default: alphabetical)
        2. Create overlapping windows of fixed size
        3. Only compare strings within same window
        
        Reduces comparisons from O(n²) to O(n*window_size)
        """
        if sort_key is None:
            sort_key = lambda x: x.lower().strip()
            
        # Sort data with original indices
        sorted_data = sorted(enumerate(data), key=lambda x: sort_key(x[1]))
        
        blocks = []
        for i in range(0, len(sorted_data), window_size // 2):  # 50% overlap
            window_end = min(i + window_size, len(sorted_data))
            block_data = sorted_data[i:window_end]
            
            blocks.append(Block(
                id=f'sorted_neighborhood_{i}',
                records=block_data,
                strategy='sorted_neighborhood',
                overlap_next=i + window_size < len(sorted_data)
            ))
            
        return blocks
        
    def create_canopy_clusters(
        self,
        data: List[str],
        loose_threshold: float = 0.6,
        tight_threshold: float = 0.8
    ) -> List[Block]:
        """
        Create blocks using canopy clustering algorithm.
        
        Algorithm:
        1. Start with random string as canopy center
        2. Add all strings within loose_threshold to canopy
        3. Remove strings within tight_threshold from remaining pool
        4. Repeat until all strings assigned to canopies
        
        Self-tuning block sizes based on data characteristics.
        """
        canopies = []
        remaining_data = list(enumerate(data))
        
        while remaining_data:
            # Select random center
            center_idx = random.randint(0, len(remaining_data) - 1)
            center = remaining_data[center_idx]
            
            # Build canopy
            canopy_members = [center]
            to_remove = []
            
            for i, (orig_idx, string) in enumerate(remaining_data):
                if i == center_idx:
                    continue
                    
                similarity = self._quick_similarity(center[1], string)
                
                if similarity >= loose_threshold:
                    canopy_members.append((orig_idx, string))
                    
                    if similarity >= tight_threshold:
                        to_remove.append(i)
            
            # Remove tightly clustered items
            remaining_data = [item for i, item in enumerate(remaining_data) 
                            if i not in to_remove and i != center_idx]
            
            canopies.append(Block(
                id=f'canopy_{len(canopies)}',
                records=canopy_members,
                strategy='canopy_clustering',
                center=center[1]
            ))
            
        return canopies
```

### Component 2: Record Linkage Integration

**Purpose**: Entity resolution and duplicate detection workflows with probabilistic matching

**Core Architecture**:
```python
class RecordLinkagePipeline:
    """
    Comprehensive record linkage pipeline integrating string similarity with probabilistic matching.
    
    Intention-Driven Interface:
    - "Find duplicate customers across databases" → multi-field probabilistic linkage
    - "Merge customer records with confidence scores" → Fellegi-Sunter model
    - "Identity resolution for marketing campaigns" → composite similarity with business rules
    
    Advanced Features:
    - Fellegi-Sunter probabilistic record linkage
    - Active learning for threshold tuning
    - Business rule integration with statistical matching
    - Confidence interval estimation for match decisions
    """
    
    def __init__(self, string_similarity: StringSimilarityPipeline):
        self.string_similarity = string_similarity
        self.probabilistic_models = ProbabilisticMatchingRegistry()
        self.active_learner = ActiveMatchingLearner()
        
    def create_linkage_composition(
        self,
        intent: str,
        source_schema: Dict,
        target_schema: Optional[Dict] = None,
        business_rules: Optional[List[BusinessRule]] = None
    ) -> LinkageComposition:
        """
        Create record linkage pipeline based on analytical intent and data schemas.
        """
        parsed_intent = self._parse_linkage_intent(intent)
        
        composition = LinkageComposition()
        
        # Field Mapping Strategy
        field_mappings = self._infer_field_mappings(source_schema, target_schema)
        
        # Comparison Vector Definition
        for field_mapping in field_mappings:
            source_field = field_mapping.source_field
            target_field = field_mapping.target_field
            field_type = field_mapping.field_type
            
            if field_type == 'name':
                composition.add_comparison(
                    field=source_field,
                    comparison_function='jaro_winkler',
                    weight=0.4,
                    m_probability=0.9,  # Prob of agreement given match
                    u_probability=0.1   # Prob of agreement given non-match
                )
            elif field_type == 'address':
                composition.add_comparison(
                    field=source_field,
                    comparison_function='address_standardized_similarity',
                    weight=0.3,
                    m_probability=0.85,
                    u_probability=0.15
                )
            elif field_type == 'date':
                composition.add_comparison(
                    field=source_field,
                    comparison_function='date_proximity',
                    weight=0.2,
                    m_probability=0.95,
                    u_probability=0.05
                )
            elif field_type == 'identifier':
                composition.add_comparison(
                    field=source_field,
                    comparison_function='exact_match',
                    weight=0.1,
                    m_probability=0.99,
                    u_probability=0.001
                )
        
        # Business Rules Integration
        if business_rules:
            for rule in business_rules:
                composition.add_business_rule(rule)
        
        # Probabilistic Model Selection
        if parsed_intent.requires_uncertainty_estimation:
            composition.set_probabilistic_model('fellegi_sunter')
        else:
            composition.set_probabilistic_model('weighted_sum')
            
        return composition

class ProbabilisticMatchingRegistry:
    """
    Registry of probabilistic record linkage models with different characteristics.
    """
    
    def __init__(self):
        self.models = {
            'fellegi_sunter': FellegiSunterModel(),
            'weighted_sum': WeightedSumModel(),
            'logistic_regression': LogisticRegressionModel(),
            'random_forest': RandomForestModel(),
            'neural_network': NeuralNetworkModel()
        }
        
    def get_model(self, model_name: str) -> ProbabilisticModel:
        return self.models[model_name]

class FellegiSunterModel:
    """
    Classic Fellegi-Sunter probabilistic record linkage model.
    
    Features:
    - EM algorithm for parameter estimation
    - Bayesian inference for match probability
    - Automatic threshold determination
    - Uncertainty quantification
    """
    
    def __init__(self):
        self.m_probabilities = {}  # P(agreement | match)
        self.u_probabilities = {}  # P(agreement | non-match)
        self.is_trained = False
        
    def train(self, comparison_vectors: np.ndarray, labels: Optional[np.ndarray] = None):
        """
        Train Fellegi-Sunter model using EM algorithm or supervised learning.
        """
        if labels is not None:
            # Supervised training
            self._train_supervised(comparison_vectors, labels)
        else:
            # Unsupervised EM training
            self._train_em(comparison_vectors)
            
        self.is_trained = True
        
    def predict_match_probability(self, comparison_vector: np.ndarray) -> MatchProbability:
        """
        Calculate match probability using Bayes theorem.
        
        P(match | agreement pattern) = 
            P(agreement pattern | match) * P(match) / P(agreement pattern)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        log_likelihood_match = 0
        log_likelihood_nonmatch = 0
        
        for field, agreement in enumerate(comparison_vector):
            field_name = f'field_{field}'
            
            if agreement:  # Fields agree
                log_likelihood_match += np.log(self.m_probabilities[field_name])
                log_likelihood_nonmatch += np.log(self.u_probabilities[field_name])
            else:  # Fields disagree
                log_likelihood_match += np.log(1 - self.m_probabilities[field_name])
                log_likelihood_nonmatch += np.log(1 - self.u_probabilities[field_name])
        
        # Convert to probabilities using logistic function
        log_odds = log_likelihood_match - log_likelihood_nonmatch
        probability = 1 / (1 + np.exp(-log_odds))
        
        return MatchProbability(
            probability=probability,
            confidence_interval=self._calculate_confidence_interval(log_odds),
            evidence_strength=abs(log_odds),
            field_contributions=self._calculate_field_contributions(comparison_vector)
        )
        
    def _train_em(self, comparison_vectors: np.ndarray, max_iterations: int = 100):
        """
        Train using Expectation-Maximization algorithm.
        """
        n_records, n_fields = comparison_vectors.shape
        
        # Initialize parameters randomly
        for field in range(n_fields):
            field_name = f'field_{field}'
            self.m_probabilities[field_name] = random.uniform(0.7, 0.95)
            self.u_probabilities[field_name] = random.uniform(0.05, 0.3)
        
        # EM iterations
        for iteration in range(max_iterations):
            # E-step: Calculate match probabilities
            match_probs = np.zeros(n_records)
            for i, vector in enumerate(comparison_vectors):
                match_prob = self.predict_match_probability(vector)
                match_probs[i] = match_prob.probability
            
            # M-step: Update parameters
            old_m_probs = self.m_probabilities.copy()
            
            for field in range(n_fields):
                field_name = f'field_{field}'
                agreements = comparison_vectors[:, field]
                
                # Update m probability
                numerator = np.sum(match_probs * agreements)
                denominator = np.sum(match_probs)
                self.m_probabilities[field_name] = numerator / denominator if denominator > 0 else 0.5
                
                # Update u probability  
                numerator = np.sum((1 - match_probs) * agreements)
                denominator = np.sum(1 - match_probs)
                self.u_probabilities[field_name] = numerator / denominator if denominator > 0 else 0.5
            
            # Check convergence
            convergence = all(
                abs(self.m_probabilities[f'field_{field}'] - old_m_probs[f'field_{field}']) < 0.001
                for field in range(n_fields)
            )
            
            if convergence:
                break
```

### Component 3: Semantic Similarity Components

**Purpose**: Context-aware text matching using embeddings and transformers

**Core Architecture**:
```python
class SemanticSimilarityPipeline:
    """
    Advanced semantic similarity pipeline using modern NLP techniques.
    
    Intention-Driven Interface:
    - "Find semantically similar product descriptions" → sentence transformers
    - "Match customer queries to FAQ answers" → BERT-based similarity
    - "Detect paraphrased content" → semantic embeddings + cosine similarity
    - "Cross-language entity matching" → multilingual embeddings
    
    Model Support:
    - Sentence-BERT for semantic text similarity
    - Word2Vec/GloVe for token-level embeddings  
    - TF-IDF for traditional text similarity
    - Custom domain-specific embedding models
    """
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.embedding_cache = EmbeddingCache()
        self.similarity_engines = self._initialize_engines()
        
    def create_semantic_composition(
        self,
        intent: str,
        text_domain: str,
        model_preferences: Optional[Dict] = None
    ) -> SemanticComposition:
        """
        Create semantic similarity pipeline based on intent and text domain.
        """
        parsed_intent = self._parse_semantic_intent(intent)
        
        composition = SemanticComposition()
        
        # Model Selection Strategy
        if text_domain == 'product_descriptions':
            composition.add_stage(
                model='sentence-transformers/all-MiniLM-L6-v2',
                similarity_function='cosine',
                weight=0.6,
                rationale="General semantic understanding for product text"
            )
            composition.add_stage(
                model='tfidf_custom',
                similarity_function='cosine',
                weight=0.4,
                rationale="Domain-specific term matching"
            )
            
        elif text_domain == 'customer_queries':
            composition.add_stage(
                model='sentence-transformers/paraphrase-MiniLM-L6-v2',
                similarity_function='cosine', 
                weight=0.7,
                rationale="Specialized for paraphrase detection"
            )
            composition.add_stage(
                model='word2vec_domain',
                similarity_function='word_mover_distance',
                weight=0.3,
                rationale="Word-level semantic similarity"
            )
            
        elif text_domain == 'scientific_papers':
            composition.add_stage(
                model='sentence-transformers/allenai-specter',
                similarity_function='cosine',
                weight=0.8,
                rationale="Scientific document understanding"
            )
            composition.add_stage(
                model='tfidf_scientific',
                similarity_function='cosine',
                weight=0.2,
                rationale="Technical term matching"
            )
        
        return composition
        
    def compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
        composition: SemanticComposition
    ) -> SemanticSimilarityResult:
        """
        Compute semantic similarity using specified composition.
        """
        stage_results = []
        
        for stage in composition.stages:
            # Get or compute embeddings
            embedding1 = self._get_cached_embedding(text1, stage.model)
            embedding2 = self._get_cached_embedding(text2, stage.model)
            
            # Compute similarity
            similarity_score = self._compute_similarity(
                embedding1, embedding2, stage.similarity_function
            )
            
            stage_results.append(StageSimilarityResult(
                stage_name=stage.model,
                similarity_score=similarity_score,
                weight=stage.weight,
                confidence=self._estimate_confidence(embedding1, embedding2, stage)
            ))
        
        # Weighted combination
        final_score = sum(
            result.similarity_score * result.weight 
            for result in stage_results
        )
        
        return SemanticSimilarityResult(
            similarity_score=final_score,
            stage_results=stage_results,
            explanation=self._generate_explanation(stage_results),
            metadata={
                'text1_length': len(text1),
                'text2_length': len(text2),
                'computation_time': time.time() - start_time
            }
        )

class ModelManager:
    """
    Manages semantic similarity models with lazy loading and caching.
    """
    
    def __init__(self):
        self.loaded_models = {}
        self.model_configs = {
            'sentence-transformers/all-MiniLM-L6-v2': {
                'type': 'sentence_transformer',
                'embedding_dim': 384,
                'max_seq_length': 256,
                'multilingual': False,
                'best_for': ['general_semantic_similarity']
            },
            'sentence-transformers/paraphrase-MiniLM-L6-v2': {
                'type': 'sentence_transformer', 
                'embedding_dim': 384,
                'max_seq_length': 128,
                'multilingual': False,
                'best_for': ['paraphrase_detection', 'duplicate_detection']
            },
            'sentence-transformers/allenai-specter': {
                'type': 'sentence_transformer',
                'embedding_dim': 768,
                'max_seq_length': 512,
                'multilingual': False,
                'best_for': ['scientific_papers', 'technical_documents']
            },
            'tfidf_custom': {
                'type': 'sklearn_tfidf',
                'max_features': 10000,
                'ngram_range': (1, 2),
                'best_for': ['keyword_matching', 'domain_specific_terms']
            }
        }
    
    def load_model(self, model_name: str):
        """Lazy load model on first use."""
        if model_name not in self.loaded_models:
            config = self.model_configs[model_name]
            
            if config['type'] == 'sentence_transformer':
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name)
                
            elif config['type'] == 'sklearn_tfidf':
                from sklearn.feature_extraction.text import TfidfVectorizer
                model = TfidfVectorizer(
                    max_features=config['max_features'],
                    ngram_range=config['ngram_range']
                )
                
            self.loaded_models[model_name] = model
            
        return self.loaded_models[model_name]
```

### Component 4: Performance Optimization for Large-Scale Processing

**Purpose**: Memory-efficient fuzzy matching with streaming and intelligent caching

**Core Architecture**:
```python
class LargeScaleFuzzyMatchingEngine:
    """
    High-performance fuzzy matching engine optimized for large datasets.
    
    Key Optimizations:
    1. Streaming similarity computation with configurable memory limits
    2. Hierarchical clustering for intelligent candidate selection
    3. Multi-level caching (embeddings, similarity matrices, phonetic codes)
    4. Parallel processing with work-stealing task distribution
    5. Progressive result refinement with early stopping
    """
    
    def __init__(self, memory_limit_mb: int = 2048):
        self.memory_limit_mb = memory_limit_mb
        self.cache_manager = MultiLevelCacheManager()
        self.parallel_executor = ParallelExecutor()
        self.progress_tracker = ProgressTracker()
        
    def process_large_dataset_matching(
        self,
        dataset: Union[pd.DataFrame, Iterator[Dict]],
        matching_composition: MatchingComposition,
        output_format: str = 'streaming'
    ) -> Union[MatchingResults, Iterator[MatchResult]]:
        """
        Process large dataset fuzzy matching with memory-efficient streaming.
        """
        # Dataset Analysis
        dataset_stats = self._analyze_dataset(dataset)
        
        # Optimization Strategy Selection
        optimization_strategy = self._select_optimization_strategy(
            dataset_stats, matching_composition
        )
        
        # Memory Management Setup
        chunk_size = self._calculate_optimal_chunk_size(
            dataset_stats, optimization_strategy
        )
        
        if output_format == 'streaming':
            return self._process_streaming(
                dataset, matching_composition, chunk_size, optimization_strategy
            )
        else:
            return self._process_batch(
                dataset, matching_composition, chunk_size, optimization_strategy
            )
    
    def _process_streaming(
        self,
        dataset: Iterator[Dict],
        composition: MatchingComposition, 
        chunk_size: int,
        optimization_strategy: OptimizationStrategy
    ) -> Iterator[MatchResult]:
        """
        Process dataset in streaming fashion with memory constraints.
        """
        current_chunk = []
        processed_count = 0
        
        for record in dataset:
            current_chunk.append(record)
            
            if len(current_chunk) >= chunk_size:
                # Process chunk
                chunk_results = self._process_chunk(
                    current_chunk, composition, optimization_strategy
                )
                
                # Yield results
                for result in chunk_results:
                    yield result
                
                # Memory management
                current_chunk.clear()
                self.cache_manager.cleanup_if_needed()
                processed_count += chunk_size
                
                # Progress reporting
                self.progress_tracker.update(processed_count)
        
        # Process final partial chunk
        if current_chunk:
            chunk_results = self._process_chunk(
                current_chunk, composition, optimization_strategy
            )
            for result in chunk_results:
                yield result

class MultiLevelCacheManager:
    """
    Multi-level caching system for different types of fuzzy matching computations.
    """
    
    def __init__(self):
        # Level 1: In-memory LRU cache for frequent lookups
        self.l1_cache = LRUCache(maxsize=10000)
        
        # Level 2: Embedding cache for semantic similarity
        self.embedding_cache = EmbeddingCache(maxsize=50000)
        
        # Level 3: Phonetic code cache for phonetic algorithms
        self.phonetic_cache = PhoneticCodeCache(maxsize=100000)
        
        # Level 4: Disk-based cache for expensive computations
        self.disk_cache = DiskCache(max_size_gb=1.0)
        
    def get_similarity(self, text1: str, text2: str, algorithm: str) -> Optional[float]:
        """Get cached similarity if available."""
        cache_key = self._generate_cache_key(text1, text2, algorithm)
        
        # Try L1 cache first
        result = self.l1_cache.get(cache_key)
        if result is not None:
            return result
            
        # Try disk cache
        result = self.disk_cache.get(cache_key)
        if result is not None:
            # Promote to L1 cache
            self.l1_cache[cache_key] = result
            return result
            
        return None
    
    def cache_similarity(self, text1: str, text2: str, algorithm: str, similarity: float):
        """Cache similarity result at appropriate level."""
        cache_key = self._generate_cache_key(text1, text2, algorithm)
        
        # Always cache in L1
        self.l1_cache[cache_key] = similarity
        
        # Cache expensive computations on disk
        if algorithm in ['semantic_similarity', 'complex_ensemble']:
            self.disk_cache[cache_key] = similarity
    
    def get_embedding(self, text: str, model: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        return self.embedding_cache.get(text, model)
    
    def cache_embedding(self, text: str, model: str, embedding: np.ndarray):
        """Cache text embedding."""
        self.embedding_cache.set(text, model, embedding)
    
    def get_phonetic_code(self, text: str, algorithm: str) -> Optional[str]:
        """Get cached phonetic code if available."""
        return self.phonetic_cache.get(text, algorithm)
    
    def cache_phonetic_code(self, text: str, algorithm: str, code: str):
        """Cache phonetic code."""
        self.phonetic_cache.set(text, algorithm, code)

class ParallelExecutor:
    """
    Parallel execution engine for fuzzy matching with work-stealing and load balancing.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, os.cpu_count())
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.work_queue = WorkStealingQueue()
        
    def parallel_similarity_computation(
        self,
        pairs: List[Tuple[str, str]],
        algorithm_composition: SimilarityComposition,
        chunk_size: int = 100
    ) -> List[SimilarityResult]:
        """
        Compute similarities for multiple pairs in parallel.
        """
        # Chunk pairs for parallel processing
        pair_chunks = [
            pairs[i:i + chunk_size] 
            for i in range(0, len(pairs), chunk_size)
        ]
        
        # Submit tasks
        futures = [
            self.executor.submit(
                self._compute_chunk_similarities,
                chunk, algorithm_composition
            )
            for chunk in pair_chunks
        ]
        
        # Collect results
        results = []
        for future in concurrent.futures.as_completed(futures):
            chunk_results = future.result()
            results.extend(chunk_results)
            
        return results
    
    def _compute_chunk_similarities(
        self,
        pair_chunk: List[Tuple[str, str]],
        composition: SimilarityComposition
    ) -> List[SimilarityResult]:
        """Compute similarities for a chunk of pairs."""
        chunk_results = []
        
        for text1, text2 in pair_chunk:
            similarity = self._compute_composed_similarity(text1, text2, composition)
            chunk_results.append(SimilarityResult(
                text1=text1,
                text2=text2,
                similarity=similarity,
                algorithm_details=composition.get_stage_results()
            ))
            
        return chunk_results
```

### Component 5: Entity Resolution Workflows

**Purpose**: Complete end-to-end entity matching pipelines with business logic integration

**Core Architecture**:
```python
class EntityResolutionWorkflow:
    """
    Complete entity resolution workflow combining all fuzzy matching components.
    
    Intention-Driven Workflows:
    - "Customer Master Data Management" → deduplication + merge + confidence scoring
    - "Product Catalog Harmonization" → cross-catalog matching + attribute reconciliation  
    - "Vendor Master Cleanup" → duplicate detection + manual review queue + auto-merge
    - "Patient Record Linkage" → privacy-preserving matching + audit trails
    
    Business Logic Integration:
    - Custom matching rules and exceptions
    - Manual review queues for uncertain matches
    - Audit trails and match provenance tracking
    - Performance monitoring and quality metrics
    """
    
    def __init__(self, 
                 string_similarity: StringSimilarityPipeline,
                 record_linkage: RecordLinkagePipeline,
                 semantic_similarity: SemanticSimilarityPipeline):
        self.string_similarity = string_similarity
        self.record_linkage = record_linkage
        self.semantic_similarity = semantic_similarity
        self.business_rules = BusinessRuleEngine()
        self.audit_logger = AuditLogger()
        
    def create_resolution_workflow(
        self,
        workflow_intent: str,
        data_sources: List[DataSource],
        business_context: Dict,
        quality_requirements: Dict
    ) -> EntityResolutionWorkflow:
        """
        Create complete entity resolution workflow based on business intent.
        """
        parsed_intent = self._parse_resolution_intent(workflow_intent)
        
        workflow = EntityResolutionWorkflow()
        
        # Data Ingestion Stage
        workflow.add_stage(DataIngestionStage(
            sources=data_sources,
            standardization_rules=self._infer_standardization_rules(data_sources),
            quality_checks=self._create_quality_checks(quality_requirements)
        ))
        
        # Blocking Stage
        workflow.add_stage(BlockingStage(
            strategy=self._select_blocking_strategy(parsed_intent, data_sources),
            performance_target=quality_requirements.get('max_execution_time')
        ))
        
        # Similarity Computation Stage
        similarity_composition = self._create_similarity_composition(
            parsed_intent, business_context
        )
        workflow.add_stage(SimilarityComputationStage(
            composition=similarity_composition,
            parallel_processing=True
        ))
        
        # Classification Stage  
        workflow.add_stage(ClassificationStage(
            model=self._select_classification_model(parsed_intent),
            thresholds=self._determine_classification_thresholds(quality_requirements),
            business_rules=self.business_rules.get_rules_for_intent(parsed_intent)
        ))
        
        # Review Queue Stage
        if quality_requirements.get('manual_review_required', False):
            workflow.add_stage(ReviewQueueStage(
                review_criteria=self._create_review_criteria(quality_requirements),
                priority_scoring=self._create_priority_scoring(business_context)
            ))
        
        # Merge/Consolidation Stage
        workflow.add_stage(ConsolidationStage(
            merge_strategy=self._select_merge_strategy(parsed_intent),
            conflict_resolution=self._create_conflict_resolution_rules(business_context),
            audit_logging=True
        ))
        
        return workflow
        
    def execute_workflow(
        self, 
        workflow: EntityResolutionWorkflow,
        monitoring_config: Optional[Dict] = None
    ) -> WorkflowExecutionResult:
        """
        Execute complete entity resolution workflow with monitoring and quality tracking.
        """
        execution_context = WorkflowExecutionContext()
        execution_results = []
        
        try:
            for stage_index, stage in enumerate(workflow.stages):
                # Pre-stage validation
                stage_validation = self._validate_stage_inputs(stage, execution_context)
                if not stage_validation.valid:
                    raise WorkflowValidationError(f"Stage {stage_index} validation failed: {stage_validation.errors}")
                
                # Execute stage
                stage_start_time = time.time()
                stage_result = self._execute_stage(stage, execution_context)
                stage_duration = time.time() - stage_start_time
                
                # Stage result validation
                result_validation = self._validate_stage_outputs(stage_result, stage.expected_outputs)
                if not result_validation.valid:
                    self._handle_stage_failure(stage, result_validation, execution_context)
                
                # Update execution context
                execution_context.add_stage_result(stage_index, stage_result)
                execution_context.update_metrics(stage.name, stage_duration, stage_result.quality_metrics)
                
                # Progress monitoring
                if monitoring_config:
                    self._report_progress(stage_index, len(workflow.stages), execution_context, monitoring_config)
                
                execution_results.append(stage_result)
                
        except Exception as e:
            self.audit_logger.log_workflow_failure(workflow.workflow_id, str(e), execution_context.get_state())
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}")
        
        # Generate final results
        final_result = WorkflowExecutionResult(
            workflow_id=workflow.workflow_id,
            execution_results=execution_results,
            quality_metrics=execution_context.get_aggregated_quality_metrics(),
            performance_metrics=execution_context.get_performance_metrics(),
            audit_trail=self.audit_logger.get_audit_trail(workflow.workflow_id),
            recommendations=self._generate_recommendations(execution_context)
        )
        
        self.audit_logger.log_workflow_completion(workflow.workflow_id, final_result)
        
        return final_result

class BusinessRuleEngine:
    """
    Business rule engine for entity resolution workflows.
    """
    
    def __init__(self):
        self.rule_registry = {}
        
    def add_rule(self, rule_name: str, rule: BusinessRule):
        """Add business rule to registry."""
        self.rule_registry[rule_name] = rule
        
    def evaluate_rules(
        self, 
        entity1: Dict, 
        entity2: Dict, 
        similarity_result: SimilarityResult,
        context: Dict
    ) -> RuleEvaluationResult:
        """Evaluate all applicable business rules."""
        rule_results = []
        
        for rule_name, rule in self.rule_registry.items():
            if rule.is_applicable(entity1, entity2, context):
                result = rule.evaluate(entity1, entity2, similarity_result, context)
                rule_results.append(RuleResult(
                    rule_name=rule_name,
                    decision=result.decision,
                    confidence=result.confidence,
                    explanation=result.explanation
                ))
        
        # Combine rule results
        final_decision = self._combine_rule_decisions(rule_results)
        
        return RuleEvaluationResult(
            decision=final_decision,
            rule_results=rule_results,
            explanation=self._generate_combined_explanation(rule_results)
        )

# Example Business Rules
class CustomerMatchingRule(BusinessRule):
    """Business rule for customer record matching."""
    
    def evaluate(self, customer1: Dict, customer2: Dict, similarity: SimilarityResult, context: Dict) -> RuleResult:
        # Rule: If SSN matches exactly but names are different, require manual review
        if (customer1.get('ssn') == customer2.get('ssn') and 
            customer1.get('ssn') is not None and
            similarity.name_similarity < 0.7):
            
            return RuleResult(
                decision='manual_review',
                confidence=0.9,
                explanation="SSN match with name discrepancy requires manual verification"
            )
        
        # Rule: High confidence match if multiple identifiers align
        identifier_matches = sum([
            customer1.get('email') == customer2.get('email') and customer1.get('email') is not None,
            customer1.get('phone') == customer2.get('phone') and customer1.get('phone') is not None,
            similarity.address_similarity > 0.8
        ])
        
        if identifier_matches >= 2:
            return RuleResult(
                decision='auto_match',
                confidence=0.95,
                explanation=f"Multiple identifier matches ({identifier_matches}) indicate high confidence match"
            )
        
        return RuleResult(
            decision='defer_to_algorithm',
            confidence=0.0,
            explanation="No specific business rule applies, use algorithmic decision"
        )
```

## Integration Architecture Summary

### Unified MCP Tool Interface
```python
@mcp.tool
def fuzzy_match_entities(
    intent: str,
    data_source: str,
    matching_fields: List[str],
    performance_requirements: Optional[Dict] = None,
    business_rules: Optional[List[str]] = None
) -> Dict:
    """
    Perform comprehensive fuzzy matching based on analytical intent.
    
    Args:
        intent: Natural language description of matching goal
        data_source: Database connection or file path
        matching_fields: Fields to use for matching
        performance_requirements: Performance constraints
        business_rules: Custom business rules to apply
    
    Returns:
        Comprehensive matching results with confidence scores and explanations
    """
    # Component Integration
    string_pipeline = StringSimilarityPipeline(memory_manager)
    linkage_pipeline = RecordLinkagePipeline(string_pipeline)  
    semantic_pipeline = SemanticSimilarityPipeline(model_manager)
    workflow_engine = EntityResolutionWorkflow(string_pipeline, linkage_pipeline, semantic_pipeline)
    
    # Create workflow from intent
    workflow = workflow_engine.create_resolution_workflow(
        workflow_intent=intent,
        data_sources=[DataSource.from_connection(data_source)],
        business_context={'matching_fields': matching_fields},
        quality_requirements=performance_requirements or {}
    )
    
    # Execute with monitoring
    result = workflow_engine.execute_workflow(workflow, monitoring_config={'progress_reporting': True})
    
    return {
        'matching_completed': True,
        'total_entities_processed': result.entities_processed,
        'matches_found': result.matches_found,
        'confidence_distribution': result.confidence_distribution,
        'execution_time_seconds': result.execution_time,
        'quality_metrics': result.quality_metrics,
        'recommended_actions': result.recommendations,
        'audit_trail_id': result.audit_trail.id
    }
```

## Key Architectural Achievements

1. **Intention-Driven Complexity**: Transforms "find duplicate customers with 85% confidence" into optimized multi-algorithm pipelines
2. **Performance at Scale**: Streaming architecture with intelligent blocking reduces O(n²) to manageable complexity
3. **Business Logic Integration**: Seamless combination of statistical matching with domain-specific rules
4. **Progressive Disclosure**: Simple string matching scales to advanced semantic similarity as needed  
5. **Rich Metadata**: Every match includes confidence scores, algorithm contributions, and explanation trails

This architecture positions LocalData MCP v2.0 as the definitive platform for fuzzy matching and entity resolution, handling everything from simple name matching to complex multi-source entity resolution workflows.