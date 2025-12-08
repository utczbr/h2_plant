#!/bin/bash
# Fix Git LFS Commit Issue

echo "Starting fix for Git LFS push error..."

# 1. Soft reset to undo the last commit but keep changes staged
echo "Step 1: Resetting last commit..."
git reset --soft HEAD~1

# 2. Configure Git LFS
echo "Step 2: Installing Git LFS hooks..."
git lfs install

# 3. Unstage the files to ensure they are picked up by LFS
echo "Step 3: Refreshing file tracking..."
git reset .

# 4. Make sure .gitattributes is tracked first
echo "Step 4: Adding .gitattributes..."
git add .gitattributes

# 5. Add the rest of the files (now LFS should pick up .pkl files)
echo "Step 5: Adding files..."
git add .

# 6. Verify LFS tracking
echo "Step 6: Verifying LFS tracking..."
LFS_FILES=$(git lfs ls-files)
if [ -z "$LFS_FILES" ]; then
    echo "WARNING: No files found in LFS! Please check if git-lfs is installed."
    echo "Files tracked:"
    git lfs ls-files
else
    echo "SUCCESS: LFS is tracking the following files:"
    git lfs ls-files
fi

echo ""
echo "Ready to commit. Run:"
echo 'git commit -m "requirements and LFS cache"'
echo 'git push'
