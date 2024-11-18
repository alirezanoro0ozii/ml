import subprocess
import sys
import os
import argparse

def git_operations(commit_message):
    try:
        # Add all changes
        subprocess.run(['git', 'add', '.'], check=True)
        print("✅ Changes added successfully")

        # Commit with provided message
        subprocess.run(['git', 'commit', '-m', commit_message], check=True)
        print(f"✅ Changes committed with message: {commit_message}")

        # Push changes
        subprocess.run(['git', 'push'], check=True)
        print("✅ Changes pushed successfully")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during git operations: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform git add, commit, and push operations')
    parser.add_argument('message', help='Commit message')
    args = parser.parse_args()

    git_operations(args.message)


# python git_ops.py "your commit message here"