#!/usr/bin/env python3
"""
Social Media Dataset Generator for LocalData MCP Stress Testing
Generates a 6GB SQLite database with realistic social media patterns:
- 25M+ posts, 5M+ users, 500M+ interactions
- Complex graph relationships (followers, mentions, shares, replies)
- Text-heavy data for testing string processing performance
- Network effects and viral content patterns
- Realistic engagement distributions and temporal patterns
"""

import os
import sqlite3
import random
import time
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Set
import math
import numpy as np
from faker import Faker
from collections import defaultdict

# Configuration constants
TARGET_SIZE_GB = 6
BYTES_PER_GB = 1024 * 1024 * 1024
TARGET_SIZE_BYTES = TARGET_SIZE_GB * BYTES_PER_GB

# Data volume targets
TARGET_USERS = 5_000_000
TARGET_POSTS = 25_000_000
TARGET_INTERACTIONS = 500_000_000

# Realistic distribution parameters
POWER_LAW_ALPHA = 2.1  # For follower distributions
VIRAL_THRESHOLD = 1000  # Interactions for viral content
ACTIVE_USER_RATIO = 0.3  # Percentage of highly active users
BOT_RATIO = 0.05  # Percentage of bot accounts

# Time simulation parameters
SIMULATION_START = datetime(2020, 1, 1)
SIMULATION_END = datetime(2024, 1, 1)
SIMULATION_DAYS = (SIMULATION_END - SIMULATION_START).days

class SocialMediaGenerator:
    """Generates realistic social media dataset with complex graph relationships."""
    
    def __init__(self, output_path: str = "social_media_test.db"):
        self.output_path = Path(output_path)
        self.fake = Faker()
        self.fake.seed_instance(12345)
        random.seed(12345)
        np.random.seed(12345)
        
        # Track generation progress
        self.current_size_bytes = 0
        self.generation_stats = {
            'users_created': 0,
            'posts_created': 0,
            'interactions_created': 0,
            'start_time': time.time()
        }
        
        # User behavior patterns
        self.user_types = {
            'influencer': 0.001,  # 0.1% - high followers, viral content
            'active': 0.099,      # 9.9% - regular posters
            'casual': 0.7,        # 70% - occasional posters
            'lurker': 0.2,        # 20% - mostly consumers
            'bot': BOT_RATIO      # 5% - automated accounts
        }
        
        # Content patterns for realistic text generation
        self.post_templates = [
            "Just had an amazing experience at {place}! {emoji} #blessed",
            "Can't believe what happened today... {story}",
            "Thoughts on {topic}? Let me know in the comments!",
            "Breaking: {news_event} - what do you think?",
            "Life update: {personal_update} {emoji}",
            "PSA: {advice} Thank me later! 😊",
            "{opinion} Anyone else agree? 🤔",
            "Throwback to when {memory}... miss those days {emoji}",
            "Currently reading: {book_title} - highly recommend!",
            "Food coma after {food}... worth it though! 🤤"
        ]
        
        self.hashtags_pool = [
            '#blessed', '#motivation', '#foodie', '#travel', '#fitness',
            '#tech', '#startup', '#entrepreneur', '#lifestyle', '#fashion',
            '#photography', '#art', '#music', '#books', '#movies',
            '#sports', '#gaming', '#coding', '#ai', '#blockchain'
        ]
        
    def create_database_schema(self) -> sqlite3.Connection:
        """Create optimized database schema for social media data."""
        print("Creating database schema...")
        
        # Remove existing database
        if self.output_path.exists():
            self.output_path.unlink()
            
        conn = sqlite3.connect(str(self.output_path))
        cursor = conn.cursor()
        
        # Enable performance optimizations
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.execute("PRAGMA synchronous = NORMAL")
        cursor.execute("PRAGMA cache_size = 100000")
        cursor.execute("PRAGMA temp_store = MEMORY")
        
        # Users table with behavioral patterns
        cursor.execute("""
            CREATE TABLE users (
                user_id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                bio TEXT,
                location TEXT,
                website TEXT,
                user_type TEXT NOT NULL, -- influencer, active, casual, lurker, bot
                follower_count INTEGER DEFAULT 0,
                following_count INTEGER DEFAULT 0,
                post_count INTEGER DEFAULT 0,
                verification_status BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL,
                last_active TIMESTAMP,
                profile_image_url TEXT,
                is_bot BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Posts table with engagement metrics
        cursor.execute("""
            CREATE TABLE posts (
                post_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_type TEXT DEFAULT 'text', -- text, image, video, link
                hashtags TEXT, -- JSON array of hashtags
                mentions TEXT, -- JSON array of mentioned user_ids
                reply_to_post_id INTEGER, -- NULL for original posts
                is_repost BOOLEAN DEFAULT FALSE,
                original_post_id INTEGER, -- For reposts
                like_count INTEGER DEFAULT 0,
                repost_count INTEGER DEFAULT 0,
                comment_count INTEGER DEFAULT 0,
                view_count INTEGER DEFAULT 0,
                is_viral BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (reply_to_post_id) REFERENCES posts (post_id),
                FOREIGN KEY (original_post_id) REFERENCES posts (post_id)
            )
        """)
        
        # Follows table for social graph
        cursor.execute("""
            CREATE TABLE follows (
                follow_id INTEGER PRIMARY KEY,
                follower_id INTEGER NOT NULL,
                following_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                UNIQUE(follower_id, following_id),
                FOREIGN KEY (follower_id) REFERENCES users (user_id),
                FOREIGN KEY (following_id) REFERENCES users (user_id)
            )
        """)
        
        # Interactions table (likes, reposts, comments)
        cursor.execute("""
            CREATE TABLE interactions (
                interaction_id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                post_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL, -- like, repost, comment, view
                content TEXT, -- For comments
                created_at TIMESTAMP NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (post_id) REFERENCES posts (post_id)
            )
        """)
        
        # Trending topics for temporal patterns
        cursor.execute("""
            CREATE TABLE trending_topics (
                topic_id INTEGER PRIMARY KEY,
                hashtag TEXT NOT NULL,
                usage_count INTEGER DEFAULT 0,
                trending_start TIMESTAMP NOT NULL,
                trending_end TIMESTAMP,
                peak_usage INTEGER DEFAULT 0,
                is_viral_event BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Create indexes for performance
        print("Creating indexes...")
        indexes = [
            "CREATE INDEX idx_users_type ON users(user_type)",
            "CREATE INDEX idx_users_created ON users(created_at)",
            "CREATE INDEX idx_posts_user ON posts(user_id)",
            "CREATE INDEX idx_posts_created ON posts(created_at)",
            "CREATE INDEX idx_posts_viral ON posts(is_viral)",
            "CREATE INDEX idx_posts_reply ON posts(reply_to_post_id)",
            "CREATE INDEX idx_follows_follower ON follows(follower_id)",
            "CREATE INDEX idx_follows_following ON follows(following_id)",
            "CREATE INDEX idx_interactions_user ON interactions(user_id)",
            "CREATE INDEX idx_interactions_post ON interactions(post_id)",
            "CREATE INDEX idx_interactions_type ON interactions(interaction_type)",
            "CREATE INDEX idx_interactions_created ON interactions(created_at)"
        ]
        
        for index in indexes:
            cursor.execute(index)
        
        conn.commit()
        return conn
    
    def generate_users(self, conn: sqlite3.Connection) -> Dict[str, List[int]]:
        """Generate users with realistic behavioral patterns."""
        print("Generating users...")
        cursor = conn.cursor()
        
        users_by_type = defaultdict(list)
        batch_size = 10000
        batch = []
        
        for user_id in range(1, TARGET_USERS + 1):
            if user_id % 100000 == 0:
                print(f"Generated {user_id:,} users...")
            
            # Determine user type based on realistic distributions
            rand = random.random()
            cumulative = 0
            user_type = 'casual'
            
            for utype, probability in self.user_types.items():
                cumulative += probability
                if rand <= cumulative:
                    user_type = utype
                    break
            
            users_by_type[user_type].append(user_id)
            
            # Generate user data
            username = self.fake.user_name() + str(random.randint(1, 9999))
            display_name = self.fake.name()
            email = f"{username}@{self.fake.free_email_domain()}"
            bio = self._generate_bio(user_type)
            location = self.fake.city() if random.random() < 0.6 else None
            website = f"https://{self.fake.domain_name()}" if random.random() < 0.1 else None
            
            # Follower counts based on user type (power law distribution)
            if user_type == 'influencer':
                follower_count = int(np.random.pareto(1.16) * 10000 + 50000)
            elif user_type == 'active':
                follower_count = int(np.random.pareto(1.5) * 500 + 100)
            elif user_type == 'bot':
                follower_count = random.randint(0, 50)
            else:  # casual, lurker
                follower_count = int(np.random.exponential(50))
            
            follower_count = min(follower_count, 10000000)  # Cap at 10M
            following_count = min(int(follower_count * random.uniform(0.1, 2.0)), 50000)
            
            verification_status = (
                user_type == 'influencer' and follower_count > 100000 and random.random() < 0.8
            )
            
            created_at = self._random_timestamp(
                SIMULATION_START, 
                SIMULATION_END - timedelta(days=30)
            )
            
            last_active = self._random_timestamp(
                created_at + timedelta(days=1),
                SIMULATION_END
            ) if user_type != 'lurker' or random.random() < 0.3 else None
            
            batch.append((
                user_id, username, display_name, email, bio, location, website,
                user_type, follower_count, following_count, 0,  # post_count updated later
                verification_status, created_at, last_active, None, user_type == 'bot'
            ))
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, batch)
                conn.commit()
                batch = []
        
        # Insert remaining batch
        if batch:
            cursor.executemany("""
                INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, batch)
            conn.commit()
        
        self.generation_stats['users_created'] = TARGET_USERS
        return users_by_type
    
    def _generate_bio(self, user_type: str) -> str:
        """Generate realistic bio based on user type."""
        bio_templates = {
            'influencer': [
                "✨ Living my best life | 📍 {location} | Collab: {email}",
                "🌟 Entrepreneur | Speaker | {follower_count}+ community",
                "💫 Inspiring others daily | Book out now! | DM for partnerships"
            ],
            'active': [
                "{hobby} enthusiast | {profession} | {location}",
                "Love {interest1}, {interest2}, and {interest3}",
                "{profession} by day, {hobby} by night"
            ],
            'casual': [
                "Just a {profession} from {location}",
                "{hobby} lover",
                "Living life one day at a time"
            ],
            'bot': [
                "Sharing daily inspiration and quotes",
                "News and updates from around the world",
                "Automated content for your timeline"
            ]
        }
        
        templates = bio_templates.get(user_type, bio_templates['casual'])
        template = random.choice(templates)
        
        return template.format(
            location=self.fake.city(),
            email=self.fake.email(),
            follower_count=random.choice(['10K', '50K', '100K', '1M']),
            hobby=random.choice(['Photography', 'Travel', 'Fitness', 'Cooking', 'Gaming']),
            profession=self.fake.job(),
            interest1=random.choice(['tech', 'art', 'music', 'sports']),
            interest2=random.choice(['food', 'travel', 'books', 'movies']),
            interest3=random.choice(['fitness', 'nature', 'photography', 'gaming'])
        )
    
    def _random_timestamp(self, start: datetime, end: datetime) -> datetime:
        """Generate random timestamp between start and end dates."""
        time_between = end - start
        days_between = time_between.days
        random_days = random.randint(0, days_between)
        random_seconds = random.randint(0, 86400)  # seconds in a day
        return start + timedelta(days=random_days, seconds=random_seconds)
    
    def generate_social_graph(self, conn: sqlite3.Connection, users_by_type: Dict[str, List[int]]):
        """Generate realistic follower relationships with network effects."""
        print("Generating social graph relationships...")
        cursor = conn.cursor()
        
        all_users = []
        for user_list in users_by_type.values():
            all_users.extend(user_list)
        
        batch_size = 10000
        batch = []
        follow_id = 1
        
        # Generate follows based on realistic patterns
        for user_type, user_ids in users_by_type.items():
            for user_id in user_ids:
                if follow_id % 100000 == 0:
                    print(f"Generated {follow_id:,} follow relationships...")
                
                # Determine following behavior based on user type
                if user_type == 'influencer':
                    follow_count = random.randint(100, 1000)  # Influencers follow many
                elif user_type == 'active':
                    follow_count = random.randint(50, 500)
                elif user_type == 'casual':
                    follow_count = random.randint(10, 100)
                elif user_type == 'lurker':
                    follow_count = random.randint(50, 200)  # Lurkers follow many but don't post
                else:  # bot
                    follow_count = random.randint(0, 20)
                
                # Select users to follow with bias toward influencers
                targets = set()
                
                # 70% chance to follow influencers
                if random.random() < 0.7 and users_by_type['influencer']:
                    influencer_follows = min(follow_count // 3, len(users_by_type['influencer']))
                    targets.update(random.sample(users_by_type['influencer'], influencer_follows))
                
                # Fill remaining with random users, avoiding self-follows
                remaining = follow_count - len(targets)
                available_users = [u for u in all_users if u != user_id and u not in targets]
                if available_users and remaining > 0:
                    additional_follows = min(remaining, len(available_users))
                    targets.update(random.sample(available_users, additional_follows))
                
                # Create follow relationships
                created_at = self._random_timestamp(SIMULATION_START, SIMULATION_END)
                for target_id in targets:
                    batch.append((follow_id, user_id, target_id, created_at))
                    follow_id += 1
                    
                    if len(batch) >= batch_size:
                        cursor.executemany("""
                            INSERT INTO follows VALUES (?,?,?,?)
                        """, batch)
                        conn.commit()
                        batch = []
        
        # Insert remaining batch
        if batch:
            cursor.executemany("""
                INSERT INTO follows VALUES (?,?,?,?)
            """, batch)
            conn.commit()
        
        print("Updating user follower counts...")
        cursor.execute("""
            UPDATE users SET following_count = (
                SELECT COUNT(*) FROM follows WHERE follower_id = users.user_id
            )
        """)
        
        cursor.execute("""
            UPDATE users SET follower_count = (
                SELECT COUNT(*) FROM follows WHERE following_id = users.user_id
            )
        """)
        
        conn.commit()
    
    def generate_posts(self, conn: sqlite3.Connection, users_by_type: Dict[str, List[int]]) -> List[int]:
        """Generate posts with realistic content and temporal patterns."""
        print("Generating posts...")
        cursor = conn.cursor()
        
        batch_size = 5000
        batch = []
        post_id = 1
        viral_posts = []
        
        # Create posting schedule with realistic temporal patterns
        posting_schedule = self._create_posting_schedule(users_by_type)
        
        for schedule_entry in posting_schedule:
            if post_id % 100000 == 0:
                print(f"Generated {post_id:,} posts...")
            
            user_id, user_type, timestamp = schedule_entry
            
            # Generate post content
            content = self._generate_post_content(user_type)
            content_type = self._select_content_type(user_type)
            hashtags = json.dumps(self._extract_hashtags(content))
            mentions = json.dumps(self._extract_mentions(content, users_by_type))
            
            # Determine if this is a reply (20% chance for active users)
            reply_to_post_id = None
            if user_type in ['active', 'casual'] and random.random() < 0.2 and post_id > 1000:
                # Reply to a recent post
                reply_to_post_id = random.randint(max(1, post_id - 10000), post_id - 1)
            
            # Generate engagement metrics (will be updated later)
            is_viral = False
            
            batch.append((
                post_id, user_id, content, content_type, hashtags, mentions,
                reply_to_post_id, False, None, 0, 0, 0, 0, is_viral, timestamp, None
            ))
            
            post_id += 1
            
            if len(batch) >= batch_size:
                cursor.executemany("""
                    INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, batch)
                conn.commit()
                batch = []
            
            if post_id > TARGET_POSTS:
                break
        
        # Insert remaining batch
        if batch:
            cursor.executemany("""
                INSERT INTO posts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, batch)
            conn.commit()
        
        self.generation_stats['posts_created'] = post_id - 1
        return viral_posts
    
    def _create_posting_schedule(self, users_by_type: Dict[str, List[int]]) -> List[Tuple[int, str, datetime]]:
        """Create realistic posting schedule with temporal patterns."""
        schedule = []
        
        for user_type, user_ids in users_by_type.items():
            # Posts per user based on type
            if user_type == 'influencer':
                posts_per_user = random.randint(1000, 5000)
            elif user_type == 'active':
                posts_per_user = random.randint(100, 500)
            elif user_type == 'casual':
                posts_per_user = random.randint(10, 50)
            elif user_type == 'bot':
                posts_per_user = random.randint(500, 2000)
            else:  # lurker
                posts_per_user = random.randint(0, 5)
            
            for user_id in user_ids:
                user_schedule = []
                
                # Generate timestamps with realistic patterns
                for _ in range(posts_per_user):
                    # Bias toward recent dates and peak hours
                    base_time = self._random_timestamp(SIMULATION_START, SIMULATION_END)
                    
                    # Adjust for peak posting hours (9 AM, 1 PM, 7 PM)
                    peak_hours = [9, 13, 19]
                    target_hour = random.choice(peak_hours)
                    
                    adjusted_time = base_time.replace(
                        hour=target_hour + random.randint(-1, 1),
                        minute=random.randint(0, 59)
                    )
                    
                    user_schedule.append((user_id, user_type, adjusted_time))
                
                schedule.extend(user_schedule)
                
                if len(schedule) >= TARGET_POSTS:
                    break
            
            if len(schedule) >= TARGET_POSTS:
                break
        
        # Sort by timestamp for realistic temporal ordering
        schedule.sort(key=lambda x: x[2])
        return schedule[:TARGET_POSTS]
    
    def _generate_post_content(self, user_type: str) -> str:
        """Generate realistic post content based on user type."""
        if user_type == 'bot':
            quotes = [
                "Believe in yourself and anything is possible! #motivation",
                "Success is not final, failure is not fatal. #wisdom",
                "The only way to do great work is to love what you do. #inspiration"
            ]
            return random.choice(quotes)
        
        template = random.choice(self.post_templates)
        
        # Fill template with realistic data
        content = template.format(
            place=self.fake.city(),
            story=self._generate_story(),
            topic=self._generate_topic(),
            news_event=self._generate_news_event(),
            personal_update=self._generate_personal_update(),
            advice=self._generate_advice(),
            opinion=self._generate_opinion(),
            memory=self._generate_memory(),
            book_title=self._generate_book_title(),
            food=self.fake.bs(),
            emoji=random.choice(['😊', '🔥', '💯', '❤️', '🙌', '✨', '💪', '🎉'])
        )
        
        # Add hashtags
        num_hashtags = random.randint(0, 5)
        hashtags = random.sample(self.hashtags_pool, min(num_hashtags, len(self.hashtags_pool)))
        if hashtags:
            content += ' ' + ' '.join(hashtags)
        
        return content
    
    def _generate_story(self) -> str:
        stories = [
            "saw the most incredible sunset",
            "met an old friend unexpectedly",
            "discovered a new favorite restaurant",
            "finished a challenging project",
            "learned something amazing today"
        ]
        return random.choice(stories)
    
    def _generate_topic(self) -> str:
        topics = [
            "remote work", "climate change", "new technology", "social media",
            "artificial intelligence", "cryptocurrency", "electric vehicles",
            "space exploration", "mental health", "sustainable living"
        ]
        return random.choice(topics)
    
    def _generate_news_event(self) -> str:
        events = [
            "Major tech company announces breakthrough",
            "New scientific discovery changes everything",
            "Celebrity couple announces engagement",
            "Sports team wins championship",
            "Economic markets hit record high"
        ]
        return random.choice(events)
    
    def _generate_personal_update(self) -> str:
        updates = [
            "started a new job",
            "moved to a new city",
            "adopted a rescue dog",
            "completed a marathon",
            "learned a new skill"
        ]
        return random.choice(updates)
    
    def _generate_advice(self) -> str:
        advice = [
            "Always backup your data regularly",
            "Drink more water throughout the day",
            "Take breaks when working from home",
            "Invest in good quality headphones",
            "Learn something new every day"
        ]
        return random.choice(advice)
    
    def _generate_opinion(self) -> str:
        opinions = [
            "Pineapple definitely belongs on pizza",
            "Morning workouts are better than evening ones",
            "Books will never be replaced by audiobooks",
            "Remote work is the future",
            "Social media breaks are essential for mental health"
        ]
        return random.choice(opinions)
    
    def _generate_memory(self) -> str:
        memories = [
            "we could travel without restrictions",
            "concerts were packed and amazing",
            "meeting friends was spontaneous",
            "life was simpler and slower",
            "technology wasn't so overwhelming"
        ]
        return random.choice(memories)
    
    def _generate_book_title(self) -> str:
        titles = [
            "The Art of Not Being Busy",
            "Digital Minimalism in Practice", 
            "The Future of Remote Work",
            "Mindfulness for Modern Life",
            "Building Sustainable Habits"
        ]
        return random.choice(titles)
    
    def _select_content_type(self, user_type: str) -> str:
        """Select content type based on user behavior."""
        if user_type == 'influencer':
            return random.choices(['text', 'image', 'video'], weights=[0.3, 0.5, 0.2])[0]
        elif user_type == 'bot':
            return 'text'
        else:
            return random.choices(['text', 'image', 'video', 'link'], weights=[0.6, 0.25, 0.1, 0.05])[0]
    
    def _extract_hashtags(self, content: str) -> List[str]:
        """Extract hashtags from post content."""
        words = content.split()
        hashtags = [word for word in words if word.startswith('#')]
        return hashtags
    
    def _extract_mentions(self, content: str, users_by_type: Dict[str, List[int]]) -> List[int]:
        """Generate realistic user mentions."""
        # Simple mention generation - could be enhanced
        if random.random() < 0.1:  # 10% chance of mentions
            all_users = []
            for user_list in users_by_type.values():
                all_users.extend(user_list)
            
            num_mentions = random.randint(1, 3)
            return random.sample(all_users, min(num_mentions, len(all_users)))
        return []
    
    def check_database_size(self) -> float:
        """Check current database size in GB."""
        if self.output_path.exists():
            size_bytes = self.output_path.stat().st_size
            size_gb = size_bytes / BYTES_PER_GB
            return size_gb
        return 0.0
    
    def generate_interactions_batch(self, conn: sqlite3.Connection, post_batch: List[int]) -> int:
        """Generate interactions for a batch of posts with realistic engagement patterns."""
        cursor = conn.cursor()
        
        interactions_created = 0
        batch_size = 10000
        batch = []
        interaction_id = 1
        
        # Get user data for engagement patterns
        cursor.execute("SELECT user_id, user_type, follower_count FROM users")
        users_data = {user_id: (user_type, follower_count) for user_id, user_type, follower_count in cursor.fetchall()}
        
        for post_id in post_batch:
            # Get post data
            cursor.execute("SELECT user_id, content, created_at FROM posts WHERE post_id = ?", (post_id,))
            post_data = cursor.fetchone()
            if not post_data:
                continue
                
            post_user_id, content, created_at = post_data
            post_user_type, post_user_followers = users_data.get(post_user_id, ('casual', 0))
            
            # Calculate engagement based on post and user characteristics
            base_engagement = self._calculate_base_engagement(post_user_type, post_user_followers, content)
            
            # Generate likes
            num_likes = int(np.random.poisson(base_engagement * 10))
            for _ in range(min(num_likes, 10000)):  # Cap to prevent memory issues
                engaging_user = self._select_engaging_user(users_data, post_user_id)
                if engaging_user:
                    interaction_time = self._random_timestamp(
                        datetime.fromisoformat(created_at.replace('T', ' ')), 
                        SIMULATION_END
                    )
                    
                    batch.append((interaction_id, engaging_user, post_id, 'like', None, interaction_time))
                    interaction_id += 1
                    interactions_created += 1
                    
                    if len(batch) >= batch_size:
                        cursor.executemany("INSERT INTO interactions VALUES (?,?,?,?,?,?)", batch)
                        conn.commit()
                        batch = []
            
            # Generate reposts (lower probability)
            num_reposts = int(np.random.poisson(base_engagement * 0.5))
            for _ in range(min(num_reposts, 1000)):
                engaging_user = self._select_engaging_user(users_data, post_user_id)
                if engaging_user:
                    interaction_time = self._random_timestamp(
                        datetime.fromisoformat(created_at.replace('T', ' ')), 
                        SIMULATION_END
                    )
                    
                    batch.append((interaction_id, engaging_user, post_id, 'repost', None, interaction_time))
                    interaction_id += 1
                    interactions_created += 1
            
            # Generate comments (even lower probability)
            num_comments = int(np.random.poisson(base_engagement * 0.2))
            for _ in range(min(num_comments, 500)):
                engaging_user = self._select_engaging_user(users_data, post_user_id)
                if engaging_user:
                    comment_content = self._generate_comment()
                    interaction_time = self._random_timestamp(
                        datetime.fromisoformat(created_at.replace('T', ' ')), 
                        SIMULATION_END
                    )
                    
                    batch.append((interaction_id, engaging_user, post_id, 'comment', comment_content, interaction_time))
                    interaction_id += 1
                    interactions_created += 1
        
        # Insert remaining batch
        if batch:
            cursor.executemany("INSERT INTO interactions VALUES (?,?,?,?,?,?)", batch)
            conn.commit()
        
        return interactions_created
    
    def _calculate_base_engagement(self, user_type: str, follower_count: int, content: str) -> float:
        """Calculate base engagement rate for a post."""
        base_rates = {
            'influencer': 5.0,
            'active': 2.0,
            'casual': 0.5,
            'lurker': 0.0,
            'bot': 0.1
        }
        
        base_rate = base_rates.get(user_type, 1.0)
        
        # Adjust for follower count (logarithmic scaling)
        follower_multiplier = math.log10(max(follower_count, 1)) / 3.0
        
        # Boost for viral content indicators
        viral_keywords = ['breaking', 'amazing', 'incredible', 'shocking', 'wow']
        if any(keyword in content.lower() for keyword in viral_keywords):
            base_rate *= 2.0
        
        # Boost for hashtag usage
        if '#' in content:
            base_rate *= 1.5
        
        return base_rate * follower_multiplier
    
    def _select_engaging_user(self, users_data: Dict[int, Tuple[str, int]], post_user_id: int) -> int:
        """Select a user to engage with the post based on realistic patterns."""
        # Exclude the post author
        available_users = [uid for uid in users_data.keys() if uid != post_user_id]
        
        if not available_users:
            return None
        
        # Weight selection by user type (active users more likely to engage)
        weights = []
        for user_id in available_users:
            user_type, _ = users_data[user_id]
            weight = {
                'influencer': 3.0,
                'active': 5.0,
                'casual': 2.0,
                'lurker': 0.5,
                'bot': 1.0
            }.get(user_type, 1.0)
            weights.append(weight)
        
        # Select user with weighted probability
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(available_users)
        
        rand = random.uniform(0, total_weight)
        cumulative = 0
        for i, weight in enumerate(weights):
            cumulative += weight
            if rand <= cumulative:
                return available_users[i]
        
        return available_users[-1]
    
    def _generate_comment(self) -> str:
        """Generate realistic comment content."""
        comments = [
            "Great post! 👍",
            "Totally agree with this!",
            "Thanks for sharing this insight",
            "This is exactly what I needed to hear today",
            "Love this perspective",
            "So true! 💯",
            "Amazing content as always",
            "This made my day 😊",
            "Couldn't agree more",
            "Well said!",
            "This is gold 🔥",
            "Needed to see this today",
            "Brilliant observation!",
            "You're absolutely right about this"
        ]
        return random.choice(comments)
    
    def update_engagement_metrics(self, conn: sqlite3.Connection):
        """Update post engagement metrics based on interactions."""
        print("Updating engagement metrics...")
        cursor = conn.cursor()
        
        # Update like counts
        cursor.execute("""
            UPDATE posts SET like_count = (
                SELECT COUNT(*) FROM interactions 
                WHERE interactions.post_id = posts.post_id AND interaction_type = 'like'
            )
        """)
        
        # Update repost counts
        cursor.execute("""
            UPDATE posts SET repost_count = (
                SELECT COUNT(*) FROM interactions 
                WHERE interactions.post_id = posts.post_id AND interaction_type = 'repost'
            )
        """)
        
        # Update comment counts
        cursor.execute("""
            UPDATE posts SET comment_count = (
                SELECT COUNT(*) FROM interactions 
                WHERE interactions.post_id = posts.post_id AND interaction_type = 'comment'
            )
        """)
        
        # Mark viral posts (high engagement)
        cursor.execute(f"""
            UPDATE posts SET is_viral = TRUE 
            WHERE like_count + repost_count + comment_count > {VIRAL_THRESHOLD}
        """)
        
        # Update user post counts
        cursor.execute("""
            UPDATE users SET post_count = (
                SELECT COUNT(*) FROM posts WHERE posts.user_id = users.user_id
            )
        """)
        
        conn.commit()
    
    def generate_dataset(self):
        """Main method to generate the complete social media dataset."""
        print(f"Starting social media dataset generation...")
        print(f"Target: {TARGET_SIZE_GB}GB database with {TARGET_USERS:,} users, {TARGET_POSTS:,} posts, {TARGET_INTERACTIONS:,} interactions")
        
        start_time = time.time()
        
        try:
            # Create database and schema
            conn = self.create_database_schema()
            
            # Generate users with behavioral patterns
            users_by_type = self.generate_users(conn)
            print(f"✓ Generated {sum(len(users) for users in users_by_type.values()):,} users")
            
            # Generate social graph relationships
            self.generate_social_graph(conn, users_by_type)
            print(f"✓ Generated social graph relationships")
            
            # Generate posts with realistic content
            viral_posts = self.generate_posts(conn, users_by_type)
            print(f"✓ Generated {self.generation_stats['posts_created']:,} posts")
            
            # Generate interactions in batches to manage memory
            print("Generating interactions...")
            cursor = conn.cursor()
            cursor.execute("SELECT post_id FROM posts ORDER BY post_id")
            all_posts = [row[0] for row in cursor.fetchall()]
            
            total_interactions = 0
            batch_size = 10000  # Process posts in batches
            
            for i in range(0, len(all_posts), batch_size):
                batch_posts = all_posts[i:i+batch_size]
                interactions_created = self.generate_interactions_batch(conn, batch_posts)
                total_interactions += interactions_created
                
                if i % (batch_size * 10) == 0:
                    current_size = self.check_database_size()
                    print(f"Processed {i + batch_size:,} posts, {total_interactions:,} interactions, DB size: {current_size:.2f}GB")
                    
                    # Stop if we've reached target size
                    if current_size >= TARGET_SIZE_GB * 0.9:  # 90% of target
                        print(f"Reached target database size, stopping generation")
                        break
            
            self.generation_stats['interactions_created'] = total_interactions
            
            # Update engagement metrics
            self.update_engagement_metrics(conn)
            
            # Final optimization
            print("Optimizing database...")
            cursor.execute("ANALYZE")
            cursor.execute("VACUUM")
            conn.commit()
            
            conn.close()
            
            # Final statistics
            final_size = self.check_database_size()
            generation_time = time.time() - start_time
            
            print("\n" + "="*60)
            print("SOCIAL MEDIA DATASET GENERATION COMPLETE")
            print("="*60)
            print(f"Database file: {self.output_path}")
            print(f"Final size: {final_size:.2f}GB")
            print(f"Generation time: {generation_time/60:.1f} minutes")
            print(f"Users created: {self.generation_stats['users_created']:,}")
            print(f"Posts created: {self.generation_stats['posts_created']:,}")
            print(f"Interactions created: {self.generation_stats['interactions_created']:,}")
            
            # Performance metrics
            posts_per_second = self.generation_stats['posts_created'] / generation_time
            interactions_per_second = self.generation_stats['interactions_created'] / generation_time
            print(f"Performance: {posts_per_second:.0f} posts/sec, {interactions_per_second:.0f} interactions/sec")
            
            return True
            
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main function to run the social media dataset generator."""
    output_dir = Path(__file__).parent.parent / "data" / "stress_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "social_media_6gb.db"
    
    generator = SocialMediaGenerator(str(output_path))
    success = generator.generate_dataset()
    
    if success:
        print(f"\n🎉 Social media dataset successfully generated!")
        print(f"Location: {output_path}")
        print(f"Ready for LocalData MCP stress testing!")
    else:
        print(f"\n❌ Dataset generation failed. Check logs for details.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())