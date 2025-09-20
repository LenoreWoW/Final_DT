#!/usr/bin/env python3
"""
Project Directory Cleanup Script
Removes temporary, duplicate, and unnecessary files while preserving essential documentation
"""

import os
import shutil
import glob
from datetime import datetime

class ProjectCleaner:
    def __init__(self, project_root="."):
        self.project_root = project_root
        self.removed_files = []
        self.kept_files = []
        self.errors = []
        
    def log(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        
    def is_essential_file(self, filepath):
        """Determine if a file is essential and should be kept."""
        
        # Essential documentation files
        essential_docs = {
            'README.md',
            'ULTIMATE_QUANTUM_PLATFORM_DOCUMENTATION.md',
            'QUANTUM_PLATFORM_PRESENTATION.md',
            'ACTUAL_TEST_RESULTS.md',
            'COMPLETE_BEGINNERS_GUIDE.md',
            'FINAL_PROJECT_SUMMARY.md',
            'PROJECT_STRUCTURE.md',
            'DOCUMENTATION_INDEX.md',
            'COMPREHENSIVE_TEST_REPORT.md',
            'REAL_IBM_QUANTUM_VERIFICATION.md'
        }
        
        # Essential configuration files
        essential_config = {
            'requirements.txt',
            '.env.example',
            '.gitignore',
            'run_app.py',
            'start.sh',
            'stop.sh'
        }
        
        # Essential test and script files
        essential_scripts = {
            'run_comprehensive_tests.py',
            'run_real_quantum_tests.py',
            'run_ibm_quantum_hardware.py'
        }
        
        # Essential PowerPoint content
        essential_powerpoint = {
            'UPDATED_POWERPOINT_CONTENT_THREE_TIER.txt'
        }
        
        filename = os.path.basename(filepath)
        
        # Check if it's an essential file
        if filename in essential_docs or filename in essential_config or filename in essential_scripts or filename in essential_powerpoint:
            return True
            
        # Keep certain directories and their contents
        essential_dirs = ['dt_project', 'tests', 'benchmark_results']
        for essential_dir in essential_dirs:
            if filepath.startswith(essential_dir + '/') or filepath.startswith('./' + essential_dir + '/'):
                return True
                
        return False
        
    def should_remove_file(self, filepath):
        """Determine if a file should be removed."""
        
        filename = os.path.basename(filepath)
        
        # Remove temporary and cache files
        temp_patterns = [
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            '*.tmp', '*.temp', '*.cache',
            '.DS_Store', 'Thumbs.db',
            '*.log', '*.pid'
        ]
        
        for pattern in temp_patterns:
            if filename.endswith(pattern.replace('*', '')) or filename == pattern:
                return True
                
        # Remove old/duplicate documentation files
        old_files = {
            'generate_powerpoint_content.py',  # Old PowerPoint generator
            'cleanup_files.py',  # Old cleanup script
            'run_ibm_quantum_tests.py',  # Superseded by newer version
            'POWERPOINT_CREATION_INSTRUCTIONS.md',  # Old instructions
            'UPDATED_POWERPOINT_CONTENT.txt',  # Old PowerPoint content (keep new THREE_TIER version)
        }
        
        if filename in old_files:
            return True
            
        # Remove files that are clearly temporary or testing
        if any(keyword in filename.lower() for keyword in ['temp', 'test_', 'debug', 'backup', 'old', 'copy']):
            # But don't remove official test files
            if not filename.startswith('test_') or 'dt_project' not in filepath:
                return True
                
        return False
        
    def clean_directory(self):
        """Clean the project directory."""
        self.log("Starting project directory cleanup...")
        
        # Get all files in project
        all_files = []
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common unneeded dirs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                filepath = os.path.join(root, file)
                all_files.append(filepath)
                
        self.log(f"Found {len(all_files)} files to analyze...")
        
        # Analyze each file
        for filepath in all_files:
            try:
                if self.should_remove_file(filepath):
                    os.remove(filepath)
                    self.removed_files.append(filepath)
                    self.log(f"Removed: {filepath}")
                elif self.is_essential_file(filepath):
                    self.kept_files.append(filepath)
                else:
                    # Non-essential but not marked for removal
                    pass
                    
            except Exception as e:
                error_msg = f"Error processing {filepath}: {str(e)}"
                self.log(error_msg)
                self.errors.append(error_msg)
                
        # Clean empty directories
        self.remove_empty_directories()
        
    def remove_empty_directories(self):
        """Remove empty directories."""
        removed_dirs = []
        
        for root, dirs, files in os.walk(self.project_root, topdown=False):
            for dirname in dirs:
                dirpath = os.path.join(root, dirname)
                try:
                    if not os.listdir(dirpath):  # Directory is empty
                        os.rmdir(dirpath)
                        removed_dirs.append(dirpath)
                        self.log(f"Removed empty directory: {dirpath}")
                except OSError:
                    pass  # Directory not empty or permission issue
                    
        return removed_dirs
        
    def generate_report(self):
        """Generate cleanup report."""
        self.log("\n" + "="*60)
        self.log("PROJECT CLEANUP COMPLETE")
        self.log("="*60)
        
        self.log(f"✅ Files removed: {len(self.removed_files)}")
        self.log(f"✅ Essential files preserved: {len(self.kept_files)}")
        
        if self.errors:
            self.log(f"⚠️ Errors encountered: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                self.log(f"   - {error}")
                
        if self.removed_files:
            self.log("\n📋 Removed files:")
            for removed in self.removed_files[:10]:  # Show first 10
                self.log(f"   - {removed}")
            if len(self.removed_files) > 10:
                self.log(f"   ... and {len(self.removed_files) - 10} more")
                
        self.log("\n📄 Essential files preserved:")
        essential_docs = [f for f in self.kept_files if f.endswith('.md')]
        for doc in essential_docs:
            self.log(f"   ✅ {doc}")
            
        self.log(f"\n🎯 Project cleanup successful!")
        self.log("Directory is now clean and organized for thesis defense.")
        
def main():
    cleaner = ProjectCleaner()
    cleaner.clean_directory()
    cleaner.generate_report()
    
if __name__ == "__main__":
    main()