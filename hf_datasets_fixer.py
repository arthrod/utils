#!/usr/bin/env python3
"""
HuggingFace Dataset Manager with AI Enhancement

This script manages HuggingFace datasets by:
1. Consolidating small datasets (<5MB) into a single repository
2. Detecting duplicate datasets based on schema (columns and row count)
3. Using AI to generate better names and READMEs for datasets
4. Providing comprehensive statistics about datasets

Duplicate detection compares:
- Column names (order-independent)
- Number of columns
- Number of rows

This finds datasets with identical structure, even if they're in different
formats or have columns in different orders.
"""

import asyncio
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import random
import time
from functools import wraps

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.columns import Columns
from rich.text import Text
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import polars as pl
import polars.selectors as cs
from huggingface_hub import (
    HfApi,
    list_datasets,
    hf_hub_download,
    upload_folder,
    create_repo,
    DatasetCard,
    DatasetCardData,
    CommitOperationDelete,
    CommitOperationAdd,
    dataset_info
)
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub.constants import ENDPOINT
import httpx
import aiohttp
from aiohttp import ClientTimeout, ClientError, ClientSession
import aiofiles

# Initialize CLI app and console
app = typer.Typer(
    name="hf-dataset-manager",
    help="""Manage and optimize HuggingFace datasets with AI enhancement.

    Features:
    • Consolidate small datasets into single repositories
    • Detect duplicates by comparing schemas (columns & rows)
    • Generate AI-powered documentation and names
    • Comprehensive dataset statistics and analytics
    • Beautiful terminal output with progress tracking
    • Async operations with 10 concurrent workers for speed

    Duplicate detection is schema-based, comparing column names
    and row counts to find structurally identical datasets.""",
    add_completion=False,
)
console = Console()
api = HfApi()

# Async HTTP Configuration
MAX_CONCURRENT_REQUESTS = 10  # Number of concurrent workers
REQUEST_TIMEOUT = 30  # Timeout per request in seconds
MAX_RETRIES = 3  # Number of retries for failed requests
RETRY_DELAY = 1  # Delay between retries in seconds

class AsyncHFClient:
    """Async client for HuggingFace Hub API with connection pooling and retry logic"""

    def __init__(self, token: Optional[str] = None, max_workers: int = MAX_CONCURRENT_REQUESTS):
        self.token = token or api.token
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        self.session: Optional[ClientSession] = None
        self.base_url = ENDPOINT
        self.timeout = ClientTimeout(total=REQUEST_TIMEOUT)

    async def __aenter__(self):
        """Create session on context entry"""
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool limit
            limit_per_host=30,  # Per-host connection limit
            ttl_dns_cache=300,  # DNS cache timeout
            enable_cleanup_closed=True
        )

        headers = {
            'User-Agent': 'hf-dataset-manager/1.0',
        }
        if self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        self.session = ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers=headers
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Close session on context exit"""
        if self.session:
            await self.session.close()
            # Small delay to allow connections to close properly
            await asyncio.sleep(0.25)

    async def fetch_with_retry(self, url: str, method: str = 'GET', **kwargs) -> Dict[str, Any]:
        """Fetch URL with retry logic and rate limiting"""
        async with self.semaphore:  # Limit concurrent requests
            for attempt in range(MAX_RETRIES):
                try:
                    async with self.session.request(method, url, **kwargs) as response:
                        response.raise_for_status()
                        return await response.json()
                except ClientError as e:
                    if attempt == MAX_RETRIES - 1:
                        console.print(f"[red]Failed after {MAX_RETRIES} attempts: {url}[/red]")
                        raise
                    await asyncio.sleep(RETRY_DELAY * (attempt + 1))  # Exponential backoff
                except Exception as e:
                    console.print(f"[yellow]Error fetching {url}: {e}[/yellow]")
                    if attempt == MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(RETRY_DELAY)

    async def get_dataset_info(self, repo_id: str) -> Dict[str, Any]:
        """Get dataset info asynchronously"""
        # Include files_metadata=true to get file sizes
        url = f"{self.base_url}/api/datasets/{repo_id}?files_metadata=true"
        try:
            data = await self.fetch_with_retry(url)
            return self._parse_dataset_info(data)
        except Exception as e:
            return {'id': repo_id, 'error': str(e), 'total_size': 0}

    def _parse_dataset_info(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse dataset info from API response"""
        details = {
            'id': data.get('id', ''),
            'author': data.get('author', None),
            'sha': data.get('sha', None),
            'created_at': data.get('createdAt', None),
            'last_modified': data.get('lastModified', None),
            'private': data.get('private', False),
            'gated': data.get('gated', False),
            'disabled': data.get('disabled', False),
            'downloads': data.get('downloads', 0) or 0,
            'downloads_all_time': data.get('downloadsAllTime', 0) or 0,
            'likes': data.get('likes', 0) or 0,
            'tags': data.get('tags', []) or [],
            'card_data': data.get('cardData', {}) or {},
            'paperswithcode_id': data.get('paperswithcode_id', None),
            'files': [],
            'total_size': 0
        }

        # Parse siblings (files) - siblings contains file information including sizes
        for file in data.get('siblings', []):
            # The size field may be None for some files
            file_size = file.get('size', 0) or 0
            details['files'].append({
                'filename': file.get('rfilename', ''),
                'size': file_size
            })
            details['total_size'] += file_size

        # Parse card data
        if details['card_data'] and isinstance(details['card_data'], dict):
            details['description'] = details['card_data'].get('description', '')
            details['citation'] = details['card_data'].get('citation', '')
            details['license'] = details['card_data'].get('license', '')
            details['language'] = details['card_data'].get('language', [])
            details['task_categories'] = details['card_data'].get('task_categories', [])

        return details

async def fetch_datasets_batch(repo_ids: List[str], progress_task=None, progress=None) -> List[Dict[str, Any]]:
    """Fetch multiple datasets concurrently using worker pool"""
    results = []

    async with AsyncHFClient() as client:
        tasks = []
        for repo_id in repo_ids:
            task = asyncio.create_task(client.get_dataset_info(repo_id))
            tasks.append(task)

        # Process results as they complete
        for future in asyncio.as_completed(tasks):
            result = await future
            results.append(result)
            if progress and progress_task is not None:
                progress.update(progress_task, advance=1)

    return results

def run_async(coro):
    """Helper to run async functions in sync context"""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new task
            task = asyncio.create_task(coro)
            return asyncio.run_coroutine_threadsafe(coro, loop).result()
        else:
            return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop, create a new one
        return asyncio.run(coro)

# Pydantic models for AI agent
class DatasetMetadata(BaseModel):
    """Metadata for a dataset"""
    name: str = Field(description="A descriptive name for the dataset")
    description: str = Field(description="A comprehensive description of the dataset")
    readme_content: str = Field(description="Full README.md content in markdown format")
    suggested_tags: List[str] = Field(description="Suggested tags for the dataset")

class DatasetAnalysis(BaseModel):
    """Analysis results for a dataset"""
    repo_id: str
    size_bytes: int
    num_rows: Optional[int]
    columns: Optional[List[str]]
    sample_data: Optional[List[Dict]]
    file_hash: Optional[str]

# AI Agent setup - Changed to gpt-5-nano (note: this model doesn't exist yet)
naming_agent = Agent(
    "openai:gpt-5-nano",  # Changed from gpt-4o-mini to gpt-5-nano
    output_type=DatasetMetadata,
    system_prompt="""You are an expert at analyzing datasets and creating comprehensive documentation.
    Given dataset information including sample data, column names, and context, you will:
    1. Generate a more descriptive and meaningful name for the dataset
    2. Write a detailed description explaining what the dataset contains and its potential uses
    3. Create a full README.md with proper markdown formatting including:
       - Dataset description
       - Data fields explanation
       - Usage examples
       - Citation information if applicable
       - License information
    4. Suggest relevant tags for better discoverability

    Make the documentation professional, informative, and helpful for potential users."""
)

def get_dataset_size(repo_id: str) -> int:
    """Get the size of a dataset in bytes"""
    try:
        # Use dataset_info with files_metadata=True to get file sizes
        dataset_info = api.dataset_info(repo_id=repo_id, files_metadata=True)
        # Get size from siblings (files in the repo)
        total_size = 0
        if hasattr(dataset_info, 'siblings') and dataset_info.siblings:
            for file in dataset_info.siblings:
                # Check if size attribute exists and is not None
                if hasattr(file, 'size') and file.size is not None:
                    total_size += file.size
        return total_size
    except Exception as e:
        console.print(f"[yellow]Warning: Could not get size for {repo_id}: {e}[/yellow]")
        return 0

def get_detailed_dataset_info(repo_id: str) -> Dict[str, Any]:
    """Get comprehensive information about a dataset"""
    try:
        # Get full dataset information including cardData and all metadata
        # IMPORTANT: Use files_metadata=True to get file sizes!
        info = api.dataset_info(repo_id=repo_id, files_metadata=True)

        details = {
            'id': info.id,
            'author': getattr(info, 'author', None),
            'sha': getattr(info, 'sha', None),
            'created_at': getattr(info, 'created_at', None),
            'last_modified': getattr(info, 'last_modified', None),
            'private': getattr(info, 'private', False),
            'gated': getattr(info, 'gated', False),
            'disabled': getattr(info, 'disabled', False),
            'downloads': getattr(info, 'downloads', 0) or 0,
            'downloads_all_time': getattr(info, 'downloads_all_time', 0) or 0,
            'likes': getattr(info, 'likes', 0) or 0,
            'tags': getattr(info, 'tags', []) or [],
            'card_data': getattr(info, 'card_data', {}) or {},
            'paperswithcode_id': getattr(info, 'paperswithcode_id', None),
            'files': [],
            'total_size': 0
        }

        # Get file information - siblings contains the files
        if hasattr(info, 'siblings') and info.siblings:
            for file in info.siblings:
                # Handle None values for size - size might be None for some files
                file_size = 0
                if hasattr(file, 'size') and file.size is not None:
                    file_size = file.size

                file_info = {
                    'filename': file.rfilename,
                    'size': file_size
                }
                details['files'].append(file_info)
                details['total_size'] += file_size

        # Get description and citation from cardData if available
        if details['card_data']:
            if isinstance(details['card_data'], dict):
                details['description'] = details['card_data'].get('description', '')
                details['citation'] = details['card_data'].get('citation', '')
                details['license'] = details['card_data'].get('license', '')
                details['language'] = details['card_data'].get('language', [])
                details['task_categories'] = details['card_data'].get('task_categories', [])
                details['task_ids'] = details['card_data'].get('task_ids', [])
                details['dataset_size'] = details['card_data'].get('dataset_size', {})

        return details

    except Exception as e:
        console.print(f"[yellow]Warning: Could not get detailed info for {repo_id}: {e}[/yellow]")
        return {
            'id': repo_id,
            'error': str(e),
            'total_size': 0,
            'downloads_all_time': 0,
            'likes': 0,
            'tags': [],
            'language': [],
            'task_categories': [],
            'license': None,
            'last_modified': None,
            'private': False,
            'gated': False
        }

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

async def analyze_dataset_lazy(repo_id: str, temp_dir: Path) -> DatasetAnalysis:
    """Analyze a dataset using Polars lazy evaluation"""
    analysis = DatasetAnalysis(
        repo_id=repo_id,
        size_bytes=get_dataset_size(repo_id),
        num_rows=None,
        columns=None,
        sample_data=None,
        file_hash=None  # Kept for backward compatibility but not used for duplicates
    )

    try:
        # Try to read dataset using Polars lazy evaluation
        dataset_path = f"hf://datasets/{repo_id}"

        # Check if dataset has parquet files
        dataset_info = api.dataset_info(repo_id)
        parquet_files = [f for f in dataset_info.siblings if f.rfilename.endswith('.parquet')]

        if parquet_files:
            # Use the first parquet file for analysis
            file_path = f"{dataset_path}/{parquet_files[0].rfilename}"

            # Use lazy evaluation to get metadata without loading full dataset
            lazy_df = pl.scan_parquet(file_path)

            # Get schema with data types (use collect_schema to avoid warning)
            schema = lazy_df.collect_schema()
            # Store columns with their types for better duplicate detection
            analysis.columns = [f"{name}:{str(dtype)}" for name, dtype in schema.items()]

            # Get row count efficiently (use pl.len() instead of deprecated pl.count())
            row_count = lazy_df.select(pl.len()).collect().item()
            analysis.num_rows = row_count

            # Get sample data (random sampling using lazy evaluation)
            if row_count > 0:
                # Sample up to 5 random rows
                sample

                analysis.sample_data = sample_df.to_dicts()

    except Exception as e:
        console.print(f"[yellow]Warning: Could not fully analyze {repo_id}: {e}[/yellow]")

    return analysis

def find_duplicates(analyses: List[DatasetAnalysis]) -> Dict[str, List[str]]:
    """Find duplicate datasets based on schema (with types) and row count"""
    # Create a signature for each dataset based on columns with types and row count
    signature_to_repos = {}

    for analysis in analyses:
        if analysis.columns and analysis.num_rows is not None:
            # Sort columns for consistent comparison
            sorted_columns = sorted(analysis.columns)
            # Create signature including column names, types, and row count
            signature = f"schema:{','.join(sorted_columns)}|rows:{analysis.num_rows}"

            if signature not in signature_to_repos:
                # Extract column names without types for display
                column_names = [col.split(':')[0] for col in sorted_columns]
                signature_to_repos[signature] = {
                    'repos': [],
                    'num_columns': len(analysis.columns),
                    'columns': column_names,
                    'columns_with_types': sorted_columns,
                    'num_rows': analysis.num_rows
                }
            signature_to_repos[signature]['repos'].append(analysis.repo_id)

    # Return only signatures with duplicates
    return {sig: info for sig, info in signature_to_repos.items() if len(info['repos']) > 1}

async def generate_improved_metadata(analysis: DatasetAnalysis) -> DatasetMetadata:
    """Use AI agent to generate improved metadata for a dataset"""
    prompt = f"""
    Analyze this dataset and create comprehensive documentation:

    Repository: {analysis.repo_id}
    Number of rows: {analysis.num_rows}
    Columns: {', '.join(analysis.columns) if analysis.columns else 'Unknown'}

    Sample data (first few rows):
    {json.dumps(analysis.sample_data, indent=2) if analysis.sample_data else 'No sample data available'}

    Please generate:
    1. A better, more descriptive name
    2. A detailed description
    3. A complete README.md with proper formatting
    4. Relevant tags for discoverability
    """

    result = await naming_agent.run(prompt)
    return result.data

def download_dataset(repo_id: str, target_dir: Path) -> Path:
    """Download a dataset to a local directory"""
    dataset_dir = target_dir / repo_id.replace("/", "_")
    dataset_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Get list of files in the dataset
        dataset_info = api.dataset_info(repo_id)

        for file_info in dataset_info.siblings:
            if file_info.rfilename != ".gitattributes":  # Skip git files
                local_file = hf_hub_download(
                    repo_id=repo_id,
                    filename=file_info.rfilename,
                    repo_type="dataset",
                    local_dir=dataset_dir
                )
                console.print(f"  Downloaded: {file_info.rfilename}")

        return dataset_dir
    except Exception as e:
        console.print(f"[red]Error downloading {repo_id}: {e}[/red]")
        return None

def create_consolidated_readme(datasets: List[str], category: str) -> str:
    """Create a README for consolidated datasets"""
    readme = f"""# Consolidated {category} Datasets

This repository contains consolidated {category} datasets for better organization and efficiency.

## Included Datasets

The following datasets have been consolidated into this repository:

"""
    for dataset in datasets:
        readme += f"- `{dataset}`\n"

    readme += f"""

## Structure

Each dataset is stored in its own subdirectory with the original structure preserved.

## Usage

To use a specific dataset from this collection:

```python
from datasets import load_dataset

# Load the entire collection
dataset = load_dataset("your-org/consolidated-{category.lower().replace(' ', '-')}")

# Access individual datasets
# Each original dataset is in its own subdirectory
```

## Metadata

- Consolidation Date: {datetime.now().isoformat()}
- Number of Datasets: {len(datasets)}
- Category: {category}

## License

Please refer to individual dataset directories for their respective licenses.
"""
    return readme

@app.command()
def analyze(
    username: str = typer.Argument(..., help="HuggingFace username to analyze"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save analysis to JSON file"),
):
    """Analyze datasets for a HuggingFace user"""
    console.print(Panel.fit(f"🔍 Analyzing datasets for user: [bold]{username}[/bold]"))

    # Get list of datasets
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching user datasets...", total=None)

        user_datasets = list(list_datasets(author=username))
        progress.update(task, completed=1)

    console.print(f"Found [green]{len(user_datasets)}[/green] datasets")

    # Analyze each dataset
    analyses = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing datasets...", total=len(user_datasets))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for dataset in user_datasets:
                analysis = asyncio.run(analyze_dataset_lazy(dataset.id, temp_path))
                analyses.append(analysis)
                progress.update(task, advance=1)

    # Display results in a table
    table = Table(title="Dataset Analysis Results")
    table.add_column("Repository", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Rows", style="green")
    table.add_column("Columns", style="yellow")

    for analysis in analyses:
        size_mb = analysis.size_bytes / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB"
        rows_str = str(analysis.num_rows) if analysis.num_rows else "N/A"
        cols_str = str(len(analysis.columns)) if analysis.columns else "N/A"

        table.add_row(analysis.repo_id, size_str, rows_str, cols_str)

    console.print(table)

    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump([a.dict() for a in analyses], f, indent=2, default=str)
        console.print(f"Analysis saved to [green]{output_file}[/green]")

@app.command()
def analyze_async(
    username: str = typer.Argument(..., help="HuggingFace username to analyze"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save analysis to JSON file"),
):
    """Analyze datasets using async operations (much faster!)"""
    console.print(Panel.fit(f"⚡ Fast Async Analysis for user: [bold]{username}[/bold]"))

    start_time = time.time()

    # Get list of datasets
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching user datasets...", total=None)
        user_datasets = list(list_datasets(author=username))
        progress.update(task, completed=1)

    console.print(f"Found [green]{len(user_datasets)}[/green] datasets")

    # Analyze each dataset asynchronously
    analyses = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• {task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Analyzing with {MAX_CONCURRENT_REQUESTS} workers...",
            total=len(user_datasets)
        )

        repo_ids = [ds.id for ds in user_datasets]
        dataset_infos = run_async(fetch_datasets_batch(repo_ids, task, progress))

        # Convert to DatasetAnalysis objects
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for info in dataset_infos:
                analysis = DatasetAnalysis(
                    repo_id=info.get('id', ''),
                    size_bytes=info.get('total_size', 0),
                    num_rows=None,  # Would need separate async call for this
                    columns=None,
                    sample_data=None,
                    file_hash=None
                )
                analyses.append(analysis)

    # Display results in a table
    table = Table(title="Dataset Analysis Results")
    table.add_column("Repository", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Downloads", style="green")
    table.add_column("Likes", style="yellow")

    for analysis, info in zip(analyses, dataset_infos):
        size_mb = analysis.size_bytes / (1024 * 1024)
        size_str = f"{size_mb:.2f} MB"
        downloads = str(info.get('downloads_all_time', 0))
        likes = str(info.get('likes', 0))

        table.add_row(analysis.repo_id, size_str, downloads, likes)

    console.print(table)

    elapsed_time = time.time() - start_time
    console.print(f"\n[green]Completed in {elapsed_time:.2f} seconds[/green]")
    console.print(f"Average: {len(user_datasets)/elapsed_time:.1f} datasets/second")

    # Save to file if requested
    if output_file:
        output_data = []
        for analysis, info in zip(analyses, dataset_infos):
            output_data.append({
                'repo_id': analysis.repo_id,
                'size_bytes': analysis.size_bytes,
                'downloads': info.get('downloads_all_time', 0),
                'likes': info.get('likes', 0),
                'tags': info.get('tags', []),
                'last_modified': info.get('last_modified', None)
            })

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        console.print(f"Analysis saved to [green]{output_file}[/green]")

    return analyses

@app.command()
def consolidate_small(
    username: str = typer.Argument(..., help="HuggingFace username"),
    organization: str = typer.Argument(..., help="Target organization for consolidated datasets"),
    size_limit_mb: float = typer.Option(5.0, "--size-limit", help="Size limit in MB"),
    delete_originals: bool = typer.Option(False, "--delete", help="Delete original datasets after consolidation"),
):
    """Consolidate small datasets into a single repository"""
    console.print(Panel.fit(f"📦 Consolidating small datasets (<{size_limit_mb}MB)"))

    # Get list of datasets
    console.print(f"Fetching datasets for user: [cyan]{username}[/cyan]")
    user_datasets = list(list_datasets(author=username))
    size_limit_bytes = size_limit_mb * 1024 * 1024

    console.print(f"Found [green]{len(user_datasets)}[/green] total datasets")
    console.print(f"Size limit: [yellow]{size_limit_mb}MB ({size_limit_bytes:,} bytes)[/yellow]")

    small_datasets = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Find small datasets
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task("Identifying small datasets...", total=len(user_datasets))

            for dataset in user_datasets:
                try:
                    # Get size of the dataset with proper API call
                    size = get_dataset_size(dataset.id)
                    if 0 < size < size_limit_bytes:  # Changed condition: size > 0 AND size < limit
                        small_datasets.append(dataset.id)
                        size_mb = size / (1024 * 1024)
                        console.print(f"  Found small dataset: [cyan]{dataset.id}[/cyan] ({size_mb:.2f}MB)")
                except Exception as e:
                    console.print(f"  [yellow]Error checking {dataset.id}: {e}[/yellow]")
                progress.update(task, advance=1)

        if not small_datasets:
            console.print("[yellow]No small datasets found[/yellow]")
            console.print("[dim]Note: Datasets must be larger than 0 bytes and smaller than the size limit[/dim]")
            return

        console.print(f"\nFound [green]{len(small_datasets)}[/green] small datasets to consolidate")

        # Download small datasets
        consolidated_dir = temp_path / "consolidated_small"
        consolidated_dir.mkdir(parents=True, exist_ok=True)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Downloading datasets...", total=len(small_datasets))

            for repo_id in small_datasets:
                console.print(f"Downloading [cyan]{repo_id}[/cyan]")
                download_dataset(repo_id, consolidated_dir)
                progress.update(task, advance=1)

        # Create consolidated repository
        consolidated_repo = f"{organization}/consolidated-small-datasets"
        console.print(f"Creating consolidated repository: [green]{consolidated_repo}[/green]")

        try:
            create_repo(consolidated_repo, repo_type="dataset", exist_ok=True)
        except Exception as e:
            console.print(f"[red]Error creating repository: {e}[/red]")
            return

        # Create README
        readme_path = consolidated_dir / "README.md"
        readme_content = create_consolidated_readme(small_datasets, "Small")
        readme_path.write_text(readme_content)

        # Create confirmation sheet
        sheet_path = consolidated_dir / "consolidation_report.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "consolidated_datasets": small_datasets,
            "total_count": len(small_datasets),
            "size_limit_mb": size_limit_mb,
            "target_repo": consolidated_repo
        }
        sheet_path.write_text(json.dumps(report, indent=2))

        # Upload to HuggingFace
        console.print("Uploading consolidated datasets...")
        upload_folder(
            folder_path=str(consolidated_dir),
            repo_id=consolidated_repo,
            repo_type="dataset",
            commit_message=f"Consolidated {len(small_datasets)} small datasets"
        )

        console.print(f"[green]✓[/green] Successfully consolidated datasets to {consolidated_repo}")

        # Delete originals if requested
        if delete_originals:
            console.print("[red]Deleting original datasets...[/red]")
            for repo_id in small_datasets:
                try:
                    api.delete_repo(repo_id=repo_id, repo_type="dataset")
                    console.print(f"  Deleted: {repo_id}")
                except Exception as e:
                    console.print(f"  [red]Failed to delete {repo_id}: {e}[/red]")

@app.command()
def handle_duplicates(
    username: str = typer.Argument(..., help="HuggingFace username"),
    organization: str = typer.Argument(..., help="Target organization for consolidated duplicates"),
    delete_duplicates: bool = typer.Option(False, "--delete", help="Delete duplicate datasets after consolidation"),
):
    """Find and consolidate duplicate datasets based on schema and row count"""
    console.print(Panel.fit("🔍 Finding and handling duplicate datasets\n[dim]Comparing columns (with types) and row counts[/dim]"))

    # Analyze datasets
    user_datasets = list(list_datasets(author=username))

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Analyze all datasets
        analyses = []
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing dataset schemas...", total=len(user_datasets))

            for dataset in user_datasets:
                analysis = asyncio.run(analyze_dataset_lazy(dataset.id, temp_path))
                analyses.append(analysis)
                progress.update(task, advance=1)

        # Find duplicates based on schema
        duplicates = find_duplicates(analyses)

        if not duplicates:
            console.print("[green]No duplicate datasets found![/green]")
            return

        console.print(f"Found [yellow]{len(duplicates)}[/yellow] groups of potential duplicate datasets\n")

        # Display duplicates with details
        duplicate_table = Table(title="Duplicate Dataset Groups")
        duplicate_table.add_column("Group", style="cyan")
        duplicate_table.add_column("Columns", style="green")
        duplicate_table.add_column("Rows", style="yellow")
        duplicate_table.add_column("Datasets", style="magenta")

        for idx, (signature, info) in enumerate(duplicates.items(), 1):
            cols_preview = ', '.join(info['columns'][:3])
            if len(info['columns']) > 3:
                cols_preview += f"... ({len(info['columns'])} total)"

            datasets_list = '\n'.join(info['repos'][:3])
            if len(info['repos']) > 3:
                datasets_list += f"\n... ({len(info['repos'])} total)"

            duplicate_table.add_row(
                f"Group {idx}",
                cols_preview,
                str(info['num_rows']),
                datasets_list
            )

        console.print(duplicate_table)

        # Consolidate duplicates
        for idx, (signature, info) in enumerate(duplicates.items(), 1):
            consolidated_dir = temp_path / f"duplicates_{idx}"
            consolidated_dir.mkdir(parents=True, exist_ok=True)

            console.print(f"\n[bold]Processing duplicate group {idx}/{len(duplicates)}[/bold]")
            console.print(f"  Schema: {info['num_columns']} columns, {info['num_rows']} rows")
            console.print(f"  Datasets: {', '.join(info['repos'])}")

            # Download all duplicates
            for repo_id in info['repos']:
                console.print(f"  Downloading [cyan]{repo_id}[/cyan]")
                download_dataset(repo_id, consolidated_dir)

            # Create consolidated repository
            consolidated_repo = f"{organization}/deduplicated-{info['num_columns']}cols-{info['num_rows']}rows-group{idx}"
            console.print(f"  Creating repository: [green]{consolidated_repo}[/green]")

            try:
                create_repo(consolidated_repo, repo_type="dataset", exist_ok=True)

                # Create detailed README
                readme_path = consolidated_dir / "README.md"
                readme_content = f"""# Deduplicated Dataset Group {idx}

## Dataset Schema Information

- **Number of Columns**: {info['num_columns']}
- **Number of Rows**: {info['num_rows']}
- **Column Names**: {', '.join(info['columns'])}

## Consolidated Datasets

The following datasets were identified as duplicates based on having identical schema (same columns and row count):

"""
                for repo in info['repos']:
                    readme_content += f"- `{repo}`\n"

                readme_content += f"""

## Consolidation Details

- **Consolidation Date**: {datetime.now().isoformat()}
- **Detection Method**: Schema comparison (column names and row count)
- **Original Count**: {len(info['repos'])} datasets

## Usage

```python
from datasets import load_dataset

# Load the consolidated dataset
dataset = load_dataset("{consolidated_repo}")
```

## Note

These datasets were identified as duplicates because they share:
1. The same number and names of columns
2. The same number of rows

Please verify the data content if exact matching is required.
"""
                readme_path.write_text(readme_content)

                # Upload
                upload_folder(
                    folder_path=str(consolidated_dir),
                    repo_id=consolidated_repo,
                    repo_type="dataset",
                    commit_message=f"Consolidated {len(info['repos'])} duplicate datasets with {info['num_columns']} columns and {info['num_rows']} rows"
                )

                console.print(f"  [green]✓[/green] Uploaded to {consolidated_repo}")

                # Delete duplicates if requested (keep the first one)
                if delete_duplicates:
                    for repo_id in info['repos'][1:]:  # Skip the first one
                        try:
                            api.delete_repo(repo_id=repo_id, repo_type="dataset")
                            console.print(f"  Deleted duplicate: {repo_id}")
                        except Exception as e:
                            console.print(f"  [red]Failed to delete {repo_id}: {e}[/red]")

            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")

@app.command()
def enhance_datasets(
    username: str = typer.Argument(..., help="HuggingFace username"),
    sample_size: int = typer.Option(3, "--sample", help="Number of datasets to enhance (for testing)"),
    commit_changes: bool = typer.Option(False, "--commit", help="Commit changes to repositories"),
):
    """Use AI to enhance dataset names and READMEs"""
    console.print(Panel.fit("🤖 Enhancing datasets with AI-generated metadata"))

    # Get datasets
    user_datasets = list(list_datasets(author=username))

    # Sample datasets for enhancement
    if sample_size < len(user_datasets):
        user_datasets = random.sample(user_datasets, sample_size)
        console.print(f"Sampling [yellow]{sample_size}[/yellow] datasets for enhancement")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        for dataset in user_datasets:
            console.print(f"\n📊 Processing [cyan]{dataset.id}[/cyan]")

            # Analyze dataset
            analysis = asyncio.run(analyze_dataset_lazy(dataset.id, temp_path))

            if analysis.columns and analysis.sample_data:
                # Generate improved metadata
                console.print("  Generating enhanced metadata with AI...")
                metadata = asyncio.run(generate_improved_metadata(analysis))

                # Display results
                console.print(f"  [green]New name:[/green] {metadata.name}")
                console.print(f"  [green]Description:[/green] {metadata.description[:200]}...")
                console.print(f"  [green]Suggested tags:[/green] {', '.join(metadata.suggested_tags)}")

                if commit_changes:
                    try:
                        # Update dataset card
                        card = DatasetCard(metadata.readme_content)
                        card.push_to_hub(dataset.id)
                        console.print(f"  [green]✓[/green] Updated README for {dataset.id}")
                    except Exception as e:
                        console.print(f"  [red]Failed to update {dataset.id}: {e}[/red]")
                else:
                    console.print("  [yellow]Dry run - no changes committed[/yellow]")
            else:
                console.print("  [yellow]Skipping - insufficient data for analysis[/yellow]")

@app.command()
def easy(
    username: str = typer.Argument(..., help="HuggingFace username"),
    organization: str = typer.Argument(..., help="Target organization for consolidated datasets"),
    delete_originals: bool = typer.Option(False, "--delete", help="Delete originals after processing"),
):
    """Run all operations: consolidate small, handle schema duplicates, and enhance remaining"""
    console.print(Panel.fit(
        "🚀 [bold]Running complete dataset optimization[/bold]\n"
        "This will:\n"
        "1. Consolidate small datasets (<5MB)\n"
        "2. Find and handle duplicate datasets (same schema)\n"
        "3. Enhance remaining datasets with AI-generated metadata",
        title="Easy Mode"
    ))

    # Step 1: Consolidate small datasets
    console.print("\n[bold]Step 1: Consolidating small datasets[/bold]")
    consolidate_small(username, organization, 5.0, delete_originals)

    # Step 2: Handle duplicates
    console.print("\n[bold]Step 2: Handling duplicate datasets[/bold]")
    handle_duplicates(username, organization, delete_originals)

    # Step 3: Enhance remaining datasets
    console.print("\n[bold]Step 3: Enhancing remaining datasets[/bold]")
    enhance_datasets(username, sample_size=5, commit_changes=True)

    console.print("\n[green]✨ Complete optimization finished![/green]")

@app.command()
def find_duplicates_only(
    username: str = typer.Argument(..., help="HuggingFace username"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Save duplicate analysis to JSON file"),
):
    """Find and display duplicate datasets without consolidating them"""
    console.print(Panel.fit("🔍 Analyzing datasets for duplicates\n[dim]Based on column structure and row count[/dim]"))

    # Get list of datasets
    user_datasets = list(list_datasets(author=username))
    console.print(f"Found [green]{len(user_datasets)}[/green] datasets to analyze\n")

    # Analyze each dataset
    analyses = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing dataset schemas...", total=len(user_datasets))

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            for dataset in user_datasets:
                analysis = asyncio.run(analyze_dataset_lazy(dataset.id, temp_path))
                analyses.append(analysis)
                progress.update(task, advance=1)

    # Find duplicates
    duplicates = find_duplicates(analyses)

    if not duplicates:
        console.print("\n[green]✨ No duplicate datasets found![/green]")
        console.print("All datasets have unique schemas (different columns or row counts).")
        return

    # Display detailed duplicate information
    console.print(f"\n[bold]Found [yellow]{len(duplicates)}[/yellow] groups of potential duplicates[/bold]\n")

    total_duplicates = sum(len(info['repos']) - 1 for info in duplicates.values())
    console.print(f"Total redundant datasets: [red]{total_duplicates}[/red]")

    # Create detailed table
    for idx, (signature, info) in enumerate(duplicates.items(), 1):
        # Create a panel for each duplicate group
        columns_display = info['columns'][:5]
        if 'columns_with_types' in info:
            # Show data types if available
            types_display = [col.split(':')[1] for col in info['columns_with_types'][:5]]
            columns_info = ', '.join([f"{name} ({dtype})" for name, dtype in zip(columns_display, types_display)])
        else:
            columns_info = ', '.join(columns_display)

        if len(info['columns']) > 5:
            columns_info += f"... ({len(info['columns'])} total)"

        group_content = f"""[bold]Schema Information:[/bold]
• Columns: {info['num_columns']}
• Rows: {info['num_rows']}
• Column details: {columns_info}

[bold]Duplicate Datasets ({len(info['repos'])} total):[/bold]"""

        for repo in info['repos']:
            size = get_dataset_size(repo)
            size_mb = size / (1024 * 1024)
            group_content += f"\n• [cyan]{repo}[/cyan] ({size_mb:.2f} MB)"

        group_content += f"\n\n[dim]Potential space saved by deduplication: {(len(info['repos']) - 1) * (size / (1024 * 1024)):.2f} MB[/dim]"

        console.print(Panel(group_content, title=f"Duplicate Group {idx}", border_style="yellow"))

    # Summary statistics
    summary_table = Table(title="Duplicate Analysis Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Total datasets analyzed", str(len(user_datasets)))
    summary_table.add_row("Duplicate groups found", str(len(duplicates)))
    summary_table.add_row("Total redundant datasets", str(total_duplicates))

    # Calculate potential space savings
    total_savings = 0
    for info in duplicates.values():
        for repo in info['repos'][1:]:  # Skip the first one (keeper)
            total_savings += get_dataset_size(repo)

    summary_table.add_row("Potential space saved", f"{total_savings / (1024 * 1024):.2f} MB")

    console.print("\n")
    console.print(summary_table)

    # Save to file if requested
    if output_file:
        output_data = {
            "analysis_date": datetime.now().isoformat(),
            "total_datasets": len(user_datasets),
            "duplicate_groups": len(duplicates),
            "total_redundant": total_duplicates,
            "potential_savings_mb": total_savings / (1024 * 1024),
            "duplicates": [
                {
                    "group_id": idx,
                    "num_columns": info['num_columns'],
                    "num_rows": info['num_rows'],
                    "columns": info['columns'],
                    "datasets": info['repos']
                }
                for idx, (_, info) in enumerate(duplicates.items(), 1)
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Analysis saved to {output_file}[/green]")

    # Provide recommendations
    console.print("\n[bold]Recommendations:[/bold]")
    console.print("• Review the duplicate groups above")
    console.print("• Run [yellow]handle-duplicates[/yellow] to consolidate them")
    console.print("• Use [yellow]--delete[/yellow] flag to remove redundant copies (keeps one)")

@app.command()
def stats(username: str = typer.Argument(..., help="HuggingFace username")):
    """Show comprehensive statistics about user's datasets (synchronous version)"""
    console.print(Panel.fit(f"📊 Comprehensive Dataset Statistics for [bold]{username}[/bold]"))

    # Get datasets with full information
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching dataset information...", total=None)

        # Get list with full metadata
        user_datasets = list(list_datasets(author=username, full=True))
        progress.update(task, completed=1)

    if not user_datasets:
        console.print("[yellow]No datasets found for this user[/yellow]")
        return

    # Calculate comprehensive statistics
    total_size = 0
    total_downloads = 0
    total_likes = 0
    size_distribution = {"<1MB": 0, "1-5MB": 0, "5-10MB": 0, "10-50MB": 0, "50-100MB": 0, ">100MB": 0}
    visibility_stats = {"public": 0, "private": 0, "gated": 0}
    tag_frequency = {}
    language_frequency = {}
    task_frequency = {}
    license_frequency = {}
    recent_datasets = []
    popular_datasets = []

    # Analyze each dataset
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Analyzing datasets...", total=len(user_datasets))

        for dataset in user_datasets:
            # Get detailed info for each dataset
            try:
                details = get_detailed_dataset_info(dataset.id)

                # Skip if there was an error
                if 'error' in details:
                    continue

                # Size statistics
                size = details.get('total_size', 0) or 0  # Handle None values
                total_size += size
                size_mb = size / (1024 * 1024)

                if size_mb < 1:
                    size_distribution["<1MB"] += 1
                elif size_mb < 5:
                    size_distribution["1-5MB"] += 1
                elif size_mb < 10:
                    size_distribution["5-10MB"] += 1
                elif size_mb < 50:
                    size_distribution["10-50MB"] += 1
                elif size_mb < 100:
                    size_distribution["50-100MB"] += 1
                else:
                    size_distribution[">100MB"] += 1

                # Download and like statistics (handle None values)
                downloads = details.get('downloads_all_time', 0) or 0
                likes = details.get('likes', 0) or 0
                total_downloads += downloads
                total_likes += likes

                # Visibility statistics
                if details.get('private', False):
                    visibility_stats["private"] += 1
                elif details.get('gated', False):
                    visibility_stats["gated"] += 1
                else:
                    visibility_stats["public"] += 1

                # Tag analysis
                for tag in details.get('tags', []):
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

                # Language analysis
                languages = details.get('language', [])
                if isinstance(languages, str):
                    languages = [languages]
                for lang in languages:
                    language_frequency[lang] = language_frequency.get(lang, 0) + 1

                # Task analysis
                for task in details.get('task_categories', []):
                    task_frequency[task] = task_frequency.get(task, 0) + 1

                # License analysis
                license_name = details.get('license', 'unknown')
                if license_name:
                    license_frequency[license_name] = license_frequency.get(license_name, 0) + 1

                # Track recent and popular datasets
                if details.get('last_modified'):
                    recent_datasets.append({
                        'id': dataset.id,
                        'modified': details['last_modified'],
                        'downloads': downloads,
                        'likes': likes
                    })

                if downloads > 0 or likes > 0:
                    popular_datasets.append({
                        'id': dataset.id,
                        'downloads': downloads,
                        'likes': likes,
                        'popularity_score': downloads + (likes * 100)  # Weight likes more
                    })

            except Exception as e:
                console.print(f"[yellow]Warning: Could not analyze {dataset.id}: {e}[/yellow]")

            progress.update(task, advance=1)

    # Display statistics (rest of the function remains the same)
    _display_statistics(
        user_datasets, total_size, total_downloads, total_likes,
        size_distribution, visibility_stats, recent_datasets,
        popular_datasets, tag_frequency, language_frequency,
        task_frequency, license_frequency
    )

@app.command()
def stats_async(username: str = typer.Argument(..., help="HuggingFace username")):
    """Show comprehensive statistics using async operations (10x faster!)"""
    console.print(Panel.fit(f"⚡ Fast Async Dataset Statistics for [bold]{username}[/bold]\n[dim]Using 10 concurrent workers[/dim]"))

    # Record start time
    start_time = time.time()

    # Get dataset list
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Fetching dataset list...", total=None)
        user_datasets = list(list_datasets(author=username))
        progress.update(task, completed=1)

    if not user_datasets:
        console.print("[yellow]No datasets found for this user[/yellow]")
        return

    console.print(f"Found [green]{len(user_datasets)}[/green] datasets to analyze")

    # Fetch all dataset info concurrently
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("• {task.completed}/{task.total} datasets"),
        TextColumn("• [cyan]{task.speed:.1f} datasets/sec[/cyan]" if hasattr('task', 'speed') else ""),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"Fetching dataset info with {MAX_CONCURRENT_REQUESTS} workers...",
            total=len(user_datasets)
        )

        # Run async fetch
        repo_ids = [ds.id for ds in user_datasets]
        all_details = run_async(fetch_datasets_batch(repo_ids, task, progress))

    # Calculate statistics
    total_size = 0
    total_downloads = 0
    total_likes = 0
    size_distribution = {"<1MB": 0, "1-5MB": 0, "5-10MB": 0, "10-50MB": 0, "50-100MB": 0, ">100MB": 0}
    visibility_stats = {"public": 0, "private": 0, "gated": 0}
    tag_frequency = {}
    language_frequency = {}
    task_frequency = {}
    license_frequency = {}
    recent_datasets = []
    popular_datasets = []

    # Process fetched data
    for details in all_details:
        if 'error' in details:
            continue

        # Size statistics
        size = details.get('total_size', 0) or 0
        total_size += size
        size_mb = size / (1024 * 1024)

        if size_mb < 1:
            size_distribution["<1MB"] += 1
        elif size_mb < 5:
            size_distribution["1-5MB"] += 1
        elif size_mb < 10:
            size_distribution["5-10MB"] += 1
        elif size_mb < 50:
            size_distribution["10-50MB"] += 1
        elif size_mb < 100:
            size_distribution["50-100MB"] += 1
        else:
            size_distribution[">100MB"] += 1

        # Download and like statistics
        downloads = details.get('downloads_all_time', 0) or 0
        likes = details.get('likes', 0) or 0
        total_downloads += downloads
        total_likes += likes

        # Visibility statistics
        if details.get('private', False):
            visibility_stats["private"] += 1
        elif details.get('gated', False):
            visibility_stats["gated"] += 1
        else:
            visibility_stats["public"] += 1

        # Tag analysis
        for tag in details.get('tags', []):
            tag_frequency[tag] = tag_frequency.get(tag, 0) + 1

        # Language analysis
        languages = details.get('language', [])
        if isinstance(languages, str):
            languages = [languages]
        for lang in languages:
            language_frequency[lang] = language_frequency.get(lang, 0) + 1

        # Task analysis
        for task in details.get('task_categories', []):
            task_frequency[task] = task_frequency.get(task, 0) + 1

        # License analysis
        license_name = details.get('license', 'unknown')
        if license_name:
            license_frequency[license_name] = license_frequency.get(license_name, 0) + 1

        # Track recent and popular datasets
        if details.get('last_modified'):
            recent_datasets.append({
                'id': details['id'],
                'modified': details['last_modified'],
                'downloads': downloads,
                'likes': likes
            })

        if downloads > 0 or likes > 0:
            popular_datasets.append({
                'id': details['id'],
                'downloads': downloads,
                'likes': likes,
                'popularity_score': downloads + (likes * 100)
            })

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Display statistics
    _display_statistics(
        user_datasets, total_size, total_downloads, total_likes,
        size_distribution, visibility_stats, recent_datasets,
        popular_datasets, tag_frequency, language_frequency,
        task_frequency, license_frequency
    )

    # Show performance improvement
    console.print("\n")
    console.print(Panel(
        f"[green]✨ Async Performance:[/green]\n"
        f"• Analyzed {len(user_datasets)} datasets in {elapsed_time:.2f} seconds\n"
        f"• Average: {len(user_datasets)/elapsed_time:.1f} datasets/second\n"
        f"• Used {MAX_CONCURRENT_REQUESTS} concurrent workers",
        title="⚡ Performance Metrics",
        border_style="green"
    ))

def _display_statistics(
    user_datasets, total_size, total_downloads, total_likes,
    size_distribution, visibility_stats, recent_datasets,
    popular_datasets, tag_frequency, language_frequency,
    task_frequency, license_frequency
):
    """Helper function to display statistics (used by both sync and async versions)"""

def _display_statistics(
    user_datasets, total_size, total_downloads, total_likes,
    size_distribution, visibility_stats, recent_datasets,
    popular_datasets, tag_frequency, language_frequency,
    task_frequency, license_frequency
):
    """Helper function to display statistics (used by both sync and async versions)"""
    # Sort recent and popular datasets
    recent_datasets = sorted(recent_datasets, key=lambda x: x['modified'], reverse=True)[:10]
    popular_datasets = sorted(popular_datasets, key=lambda x: x['popularity_score'], reverse=True)[:10]

    # Display comprehensive statistics
    console.print("\n")

    # Overview Panel - handle division by zero
    avg_size_mb = (total_size / len(user_datasets)) / (1024 * 1024) if user_datasets else 0
    avg_downloads = total_downloads // len(user_datasets) if user_datasets else 0
    avg_likes = total_likes / len(user_datasets) if user_datasets else 0

    overview_text = f"""[bold]Total Datasets:[/bold] {len(user_datasets)}
[bold]Total Size:[/bold] {total_size / (1024 * 1024):.2f} MB ({total_size / (1024 * 1024 * 1024):.2f} GB)
[bold]Average Size:[/bold] {avg_size_mb:.2f} MB
[bold]Total Downloads:[/bold] {total_downloads:,}
[bold]Total Likes:[/bold] {total_likes}
[bold]Average Downloads:[/bold] {avg_downloads:,}
[bold]Average Likes:[/bold] {avg_likes:.1f}"""

    console.print(Panel(overview_text, title="📈 Overview", border_style="green"))

    # Size Distribution
    console.print("\n[bold]📦 Size Distribution:[/bold]")
    for size_range, count in size_distribution.items():
        if count > 0:
            percentage = (count / len(user_datasets)) * 100
            bar = "█" * int(percentage / 2)
            console.print(f"  {size_range:>10}: {bar} {count} ({percentage:.1f}%)")

    # Visibility Statistics
    console.print("\n[bold]👁️ Visibility:[/bold]")
    for vis_type, count in visibility_stats.items():
        if count > 0:
            percentage = (count / len(user_datasets)) * 100
            console.print(f"  {vis_type.capitalize():>10}: {count} datasets ({percentage:.1f}%)")

    # Most Popular Datasets
    if popular_datasets:
        pop_table = Table(title="🌟 Most Popular Datasets")
        pop_table.add_column("Dataset", style="cyan")
        pop_table.add_column("Downloads", style="green", justify="right")
        pop_table.add_column("Likes", style="yellow", justify="right")

        for ds in popular_datasets[:5]:
            pop_table.add_row(
                ds['id'].split('/')[-1] if '/' in ds['id'] else ds['id'],
                f"{ds['downloads']:,}",
                str(ds['likes'])
            )

        console.print("\n")
        console.print(pop_table)

    # Recently Updated Datasets
    if recent_datasets:
        recent_table = Table(title="🕐 Recently Updated Datasets")
        recent_table.add_column("Dataset", style="cyan")
        recent_table.add_column("Last Modified", style="magenta")
        recent_table.add_column("Downloads", style="green", justify="right")

        for ds in recent_datasets[:5]:
            # Format the date
            if ds['modified']:
                try:
                    mod_date = datetime.fromisoformat(ds['modified'].replace('Z', '+00:00'))
                    days_ago = (datetime.now(mod_date.tzinfo) - mod_date).days
                    if days_ago == 0:
                        time_str = "Today"
                    elif days_ago == 1:
                        time_str = "Yesterday"
                    elif days_ago < 7:
                        time_str = f"{days_ago} days ago"
                    elif days_ago < 30:
                        time_str = f"{days_ago // 7} weeks ago"
                    else:
                        time_str = f"{days_ago // 30} months ago"
                except:
                    time_str = str(ds['modified'])[:10]
            else:
                time_str = "Unknown"

            recent_table.add_row(
                ds['id'].split('/')[-1] if '/' in ds['id'] else ds['id'],
                time_str,
                f"{ds['downloads']:,}"
            )

        console.print("\n")
        console.print(recent_table)

    # Top Tags
    if tag_frequency:
        console.print("\n[bold]🏷️ Top Tags:[/bold]")
        sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        for tag, count in sorted_tags:
            console.print(f"  • {tag}: {count} datasets")

    # Top Languages
    if language_frequency:
        console.print("\n[bold]🌍 Top Languages:[/bold]")
        sorted_langs = sorted(language_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        for lang, count in sorted_langs:
            console.print(f"  • {lang}: {count} datasets")

    # License Distribution
    if license_frequency:
        console.print("\n[bold]📄 License Distribution:[/bold]")
        sorted_licenses = sorted(license_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        for license_name, count in sorted_licenses:
            percentage = (count / len(user_datasets)) * 100
            console.print(f"  • {license_name}: {count} ({percentage:.1f}%)")

    # Task Categories
    if task_frequency:
        console.print("\n[bold]🎯 Task Categories:[/bold]")
        sorted_tasks = sorted(task_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        for task, count in sorted_tasks:
            console.print(f"  • {task}: {count} datasets")

    # Summary Panel
    summary_text = f"""[green]✅ Analysis Complete![/green]

Use [yellow]analyze[/yellow] to get detailed information about each dataset
Use [yellow]find-duplicates-only[/yellow] to identify duplicate datasets
Use [yellow]consolidate-small[/yellow] to optimize storage of small datasets
Use [yellow]enhance-datasets[/yellow] to improve documentation with AI
Try [yellow]stats-async[/yellow] for 10x faster analysis with concurrent workers!"""

    console.print("\n")
    console.print(Panel(summary_text, title="💡 Next Steps", border_style="blue"))

@app.command()
def benchmark(
    username: str = typer.Argument(..., help="HuggingFace username"),
    sample_size: int = typer.Option(20, "--sample", help="Number of datasets to benchmark")
):
    """Benchmark sync vs async performance"""
    console.print(Panel.fit("⚡ Performance Benchmark: Sync vs Async"))

    # Get dataset list
    user_datasets = list(list_datasets(author=username))[:sample_size]

    if not user_datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return

    console.print(f"Benchmarking with [cyan]{len(user_datasets)}[/cyan] datasets\n")

    # Test synchronous version
    console.print("[bold]Testing synchronous version...[/bold]")
    sync_start = time.time()

    for dataset in user_datasets:
        try:
            _ = get_detailed_dataset_info(dataset.id)
        except:
            pass

    sync_time = time.time() - sync_start

    # Test async version
    console.print("[bold]Testing async version with 10 workers...[/bold]")
    async_start = time.time()

    repo_ids = [ds.id for ds in user_datasets]
    _ = run_async(fetch_datasets_batch(repo_ids))

    async_time = time.time() - async_start

    # Display results
    speedup = sync_time / async_time

    results_table = Table(title="📊 Benchmark Results")
    results_table.add_column("Method", style="cyan")
    results_table.add_column("Time (seconds)", style="yellow")
    results_table.add_column("Datasets/second", style="green")

    results_table.add_row(
        "Synchronous",
        f"{sync_time:.2f}",
        f"{len(user_datasets)/sync_time:.2f}"
    )
    results_table.add_row(
        "Async (10 workers)",
        f"{async_time:.2f}",
        f"{len(user_datasets)/async_time:.2f}"
    )

    console.print("\n")
    console.print(results_table)

    console.print("\n")
    console.print(Panel(
        f"[green]✨ Async is {speedup:.1f}x faster![/green]\n"
        f"Time saved: {sync_time - async_time:.2f} seconds",
        title="🎉 Results",
        border_style="green"
    ))

if __name__ == "__main__":
    app()
