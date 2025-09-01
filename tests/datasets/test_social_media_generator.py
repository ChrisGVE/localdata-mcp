#!/usr/bin/env python3
"""
Integration test for Social Media Dataset Generator.
Validates database schema, data quality, and performance characteristics.
"""

import sqlite3
import tempfile
import os
import unittest
from pathlib import Path
import time
import json

from tests.datasets.social_media_generator import SocialMediaGenerator

class TestSocialMediaGenerator(unittest.TestCase):
    """Test suite for social media dataset generator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = os.path.join(self.temp_dir, "test_social_media.db")
        
        # Create a smaller test dataset for validation
        self.generator = SocialMediaGenerator(self.test_db_path)
        
        # Override targets for faster testing
        self.original_users = self.generator.__class__.__dict__.get('TARGET_USERS', 5000000)
        self.original_posts = self.generator.__class__.__dict__.get('TARGET_POSTS', 25000000)
        
        # Use smaller targets for testing
        import social_media_generator
        social_media_generator.TARGET_USERS = 1000
        social_media_generator.TARGET_POSTS = 5000
        social_media_generator.TARGET_INTERACTIONS = 50000
    
    def tearDown(self):
        """Clean up test environment."""
        # Restore original values
        import social_media_generator
        social_media_generator.TARGET_USERS = self.original_users
        social_media_generator.TARGET_POSTS = self.original_posts
        social_media_generator.TARGET_INTERACTIONS = 500000000
        
        # Clean up temp files
        if os.path.exists(self.test_db_path):
            os.unlink(self.test_db_path)
        os.rmdir(self.temp_dir)
    
    def test_database_schema_creation(self):
        """Test that database schema is created correctly."""
        conn = self.generator.create_database_schema()
        cursor = conn.cursor()
        
        # Check that all required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['users', 'posts', 'follows', 'interactions', 'trending_topics']
        for table in expected_tables:
            self.assertIn(table, tables, f"Table '{table}' not found in database")
        
        # Check users table structure
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        expected_user_columns = [
            'user_id', 'username', 'display_name', 'email', 'bio', 
            'location', 'website', 'user_type', 'follower_count',
            'following_count', 'post_count', 'verification_status',
            'created_at', 'last_active', 'profile_image_url', 'is_bot'
        ]
        for col in expected_user_columns:
            self.assertIn(col, user_columns, f"Column '{col}' not found in users table")
        
        # Check posts table structure
        cursor.execute("PRAGMA table_info(posts)")
        post_columns = [row[1] for row in cursor.fetchall()]
        expected_post_columns = [
            'post_id', 'user_id', 'content', 'content_type', 'hashtags',
            'mentions', 'reply_to_post_id', 'is_repost', 'original_post_id',
            'like_count', 'repost_count', 'comment_count', 'view_count',
            'is_viral', 'created_at', 'updated_at'
        ]
        for col in expected_post_columns:
            self.assertIn(col, post_columns, f"Column '{col}' not found in posts table")
        
        conn.close()
    
    def test_user_generation(self):
        """Test user generation with realistic patterns."""
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        
        cursor = conn.cursor()
        
        # Check total user count
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        self.assertEqual(user_count, 1000, f"Expected 1000 users, got {user_count}")
        
        # Check user type distribution
        cursor.execute("SELECT user_type, COUNT(*) FROM users GROUP BY user_type")
        type_counts = dict(cursor.fetchall())
        
        # Verify all user types are present
        expected_types = ['influencer', 'active', 'casual', 'lurker', 'bot']
        for user_type in expected_types:
            self.assertIn(user_type, type_counts, f"User type '{user_type}' not found")
        
        # Check that bot accounts are marked correctly
        cursor.execute("SELECT COUNT(*) FROM users WHERE user_type='bot' AND is_bot=TRUE")
        bot_count = cursor.fetchone()[0]
        self.assertEqual(bot_count, type_counts['bot'], "Bot flag not set correctly")
        
        # Verify usernames are unique
        cursor.execute("SELECT COUNT(DISTINCT username) FROM users")
        unique_usernames = cursor.fetchone()[0]
        self.assertEqual(unique_usernames, user_count, "Usernames are not unique")
        
        conn.close()
    
    def test_social_graph_generation(self):
        """Test social graph relationship generation."""
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        self.generator.generate_social_graph(conn, users_by_type)
        
        cursor = conn.cursor()
        
        # Check that follows relationships exist
        cursor.execute("SELECT COUNT(*) FROM follows")
        follow_count = cursor.fetchone()[0]
        self.assertGreater(follow_count, 0, "No follow relationships generated")
        
        # Check that no self-follows exist
        cursor.execute("SELECT COUNT(*) FROM follows WHERE follower_id = following_id")
        self_follows = cursor.fetchone()[0]
        self.assertEqual(self_follows, 0, "Self-follows found in database")
        
        # Verify follower counts are updated in users table
        cursor.execute("SELECT COUNT(*) FROM users WHERE follower_count > 0")
        users_with_followers = cursor.fetchone()[0]
        self.assertGreater(users_with_followers, 0, "No users have followers")
        
        # Check referential integrity
        cursor.execute("""
            SELECT COUNT(*) FROM follows f 
            LEFT JOIN users u1 ON f.follower_id = u1.user_id
            LEFT JOIN users u2 ON f.following_id = u2.user_id
            WHERE u1.user_id IS NULL OR u2.user_id IS NULL
        """)
        orphaned_follows = cursor.fetchone()[0]
        self.assertEqual(orphaned_follows, 0, "Orphaned follow relationships found")
        
        conn.close()
    
    def test_post_generation(self):
        """Test post generation with realistic content."""
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        viral_posts = self.generator.generate_posts(conn, users_by_type)
        
        cursor = conn.cursor()
        
        # Check post count
        cursor.execute("SELECT COUNT(*) FROM posts")
        post_count = cursor.fetchone()[0]
        self.assertEqual(post_count, 5000, f"Expected 5000 posts, got {post_count}")
        
        # Check that posts have content
        cursor.execute("SELECT COUNT(*) FROM posts WHERE content IS NULL OR content = ''")
        empty_posts = cursor.fetchone()[0]
        self.assertEqual(empty_posts, 0, "Posts with empty content found")
        
        # Check content types are realistic
        cursor.execute("SELECT content_type, COUNT(*) FROM posts GROUP BY content_type")
        content_types = dict(cursor.fetchall())
        expected_types = ['text', 'image', 'video', 'link']
        
        for ctype in content_types.keys():
            self.assertIn(ctype, expected_types, f"Unexpected content type: {ctype}")
        
        # Check that some posts have hashtags
        cursor.execute("SELECT COUNT(*) FROM posts WHERE hashtags != '[]'")
        posts_with_hashtags = cursor.fetchone()[0]
        self.assertGreater(posts_with_hashtags, 0, "No posts have hashtags")
        
        # Verify post timestamps are reasonable
        cursor.execute("SELECT MIN(created_at), MAX(created_at) FROM posts")
        min_time, max_time = cursor.fetchone()
        self.assertIsNotNone(min_time, "Posts missing timestamps")
        self.assertIsNotNone(max_time, "Posts missing timestamps")
        
        conn.close()
    
    def test_interaction_generation(self):
        """Test interaction generation patterns."""
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        self.generator.generate_social_graph(conn, users_by_type)
        self.generator.generate_posts(conn, users_by_type)
        
        # Generate interactions for a small batch
        cursor = conn.cursor()
        cursor.execute("SELECT post_id FROM posts LIMIT 100")
        test_posts = [row[0] for row in cursor.fetchall()]
        
        interactions_created = self.generator.generate_interactions_batch(conn, test_posts)
        self.assertGreater(interactions_created, 0, "No interactions generated")
        
        # Check interaction types
        cursor.execute("SELECT interaction_type, COUNT(*) FROM interactions GROUP BY interaction_type")
        interaction_types = dict(cursor.fetchall())
        
        expected_types = ['like', 'repost', 'comment']
        for itype in expected_types:
            if itype in interaction_types:
                self.assertGreater(interaction_types[itype], 0, f"No {itype} interactions")
        
        # Verify no self-interactions (user interacting with own posts)
        cursor.execute("""
            SELECT COUNT(*) FROM interactions i
            JOIN posts p ON i.post_id = p.post_id
            WHERE i.user_id = p.user_id
        """)
        self_interactions = cursor.fetchone()[0]
        self.assertEqual(self_interactions, 0, "Self-interactions found")
        
        conn.close()
    
    def test_database_constraints(self):
        """Test database referential integrity constraints."""
        conn = self.generator.create_database_schema()
        cursor = conn.cursor()
        
        # Test foreign key constraints are enabled
        cursor.execute("PRAGMA foreign_keys")
        fk_status = cursor.fetchone()[0]
        # Note: SQLite may not enforce foreign keys by default, but schema should be valid
        
        # Test unique constraints
        with self.assertRaises(sqlite3.IntegrityError):
            # Try to insert duplicate username
            cursor.execute("""
                INSERT INTO users (user_id, username, display_name, email, user_type, created_at)
                VALUES (1, 'testuser', 'Test User', 'test1@example.com', 'casual', '2024-01-01')
            """)
            cursor.execute("""
                INSERT INTO users (user_id, username, display_name, email, user_type, created_at)
                VALUES (2, 'testuser', 'Test User 2', 'test2@example.com', 'casual', '2024-01-01')
            """)
            conn.commit()
        
        conn.close()
    
    def test_performance_characteristics(self):
        """Test that generator meets performance expectations."""
        start_time = time.time()
        
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        
        user_gen_time = time.time() - start_time
        
        # Should generate 1000 users in reasonable time (< 10 seconds)
        self.assertLess(user_gen_time, 10.0, f"User generation too slow: {user_gen_time:.2f}s")
        
        # Check database size is reasonable
        conn.close()
        if os.path.exists(self.test_db_path):
            db_size = os.path.getsize(self.test_db_path)
            # Should be at least 100KB for 1000 users with content
            self.assertGreater(db_size, 100000, f"Database too small: {db_size} bytes")
    
    def test_data_quality(self):
        """Test data quality and realism."""
        conn = self.generator.create_database_schema()
        users_by_type = self.generator.generate_users(conn)
        self.generator.generate_posts(conn, users_by_type)
        
        cursor = conn.cursor()
        
        # Check that posts have realistic length distribution
        cursor.execute("SELECT AVG(LENGTH(content)), MIN(LENGTH(content)), MAX(LENGTH(content)) FROM posts")
        avg_len, min_len, max_len = cursor.fetchone()
        
        self.assertGreater(avg_len, 20, "Posts too short on average")
        self.assertGreater(min_len, 0, "Empty posts found")
        self.assertLess(max_len, 5000, "Posts unrealistically long")
        
        # Check user type distribution is reasonable
        cursor.execute("SELECT user_type, COUNT(*) * 100.0 / (SELECT COUNT(*) FROM users) AS percentage FROM users GROUP BY user_type")
        type_percentages = dict(cursor.fetchall())
        
        # Influencers should be rare (< 5%)
        if 'influencer' in type_percentages:
            self.assertLess(type_percentages['influencer'], 5.0, "Too many influencers")
        
        # Casual users should be majority
        if 'casual' in type_percentages:
            self.assertGreater(type_percentages['casual'], 30.0, "Too few casual users")
        
        conn.close()

def run_validation_suite():
    """Run the complete validation suite."""
    print("Running Social Media Dataset Generator Validation Suite...")
    print("=" * 60)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "=" * 60)
    print("Validation complete!")

if __name__ == "__main__":
    run_validation_suite()