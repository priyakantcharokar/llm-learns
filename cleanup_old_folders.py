#!/usr/bin/env python3
"""
Cleanup script to remove old l1 and l2 folders and all __pycache__ directories.
Run this script to clean up the project.
"""
import os
import shutil

def cleanup_project():
    """Remove old folders and cache files."""
    print("Cleaning up project...")
    
    # Remove old l1 and l2 folders
    folders_to_remove = ['l1', 'l2']
    for folder in folders_to_remove:
        if os.path.exists(folder):
            print(f"Removing {folder}/ folder...")
            try:
                shutil.rmtree(folder)
                print(f"✓ Successfully removed {folder}/")
            except Exception as e:
                print(f"✗ Error removing {folder}/: {e}")
        else:
            print(f"  {folder}/ does not exist, skipping...")
    
    # Remove all __pycache__ directories
    print("\nRemoving __pycache__ directories...")
    removed_count = 0
    for root, dirs, files in os.walk('.'):
        # Skip .git and other hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        if '__pycache__' in dirs:
            cache_dir = os.path.join(root, '__pycache__')
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ Removed {cache_dir}")
                removed_count += 1
            except Exception as e:
                print(f"✗ Error removing {cache_dir}: {e}")
    
    print(f"\n✓ Cleanup complete! Removed {removed_count} __pycache__ directories.")

if __name__ == "__main__":
    cleanup_project()

