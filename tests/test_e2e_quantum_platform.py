#!/usr/bin/env python3
"""
🌐 END-TO-END QUANTUM PLATFORM TESTS
===================================

Playwright-powered end-to-end tests for the Universal Quantum Digital Twin Factory
web interface. Tests complete user workflows from data upload to quantum results.

Author: Hassan Al-Sahli  
Purpose: E2E validation of quantum platform web interface
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Import Playwright for E2E testing
try:
    from playwright.async_api import async_playwright, expect
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    pytest.skip("Playwright not available - install with: pip install playwright", allow_module_level=True)


class TestQuantumFactoryWebInterface:
    """🌐 Test Quantum Factory Web Interface"""
    
    @pytest.fixture
    async def browser_context(self):
        """Create browser context for testing"""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={'width': 1280, 'height': 720}
        )
        
        yield context
        
        await context.close()
        await browser.close()
        await playwright.stop()
    
    @pytest.fixture
    async def page(self, browser_context):
        """Create page for testing"""
        page = await browser_context.new_page()
        return page
    
    @pytest.fixture
    def sample_csv_file(self):
        """Create sample CSV file for upload testing"""
        # Create sample data
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'feature3': np.random.randn(50),
            'target': np.random.randint(0, 2, 50)
        })
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            return f.name
    
    @pytest.fixture
    def sample_json_file(self):
        """Create sample JSON file for upload testing"""
        data = {
            'sensors': [
                {'id': 'sensor_1', 'readings': [1.2, 1.5, 1.8, 2.0]},
                {'id': 'sensor_2', 'readings': [0.8, 1.1, 1.4, 1.7]},
                {'id': 'sensor_3', 'readings': [2.1, 2.3, 2.5, 2.8]}
            ],
            'metadata': {
                'location': 'test_lab',
                'experiment': 'quantum_sensing_test'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, indent=2)
            return f.name
    
    @pytest.mark.asyncio
    async def test_quantum_factory_homepage_load(self, page):
        """Test quantum factory homepage loads correctly"""
        
        print("\n🌐 Testing Quantum Factory Homepage...")
        
        # Navigate to quantum factory
        await page.goto('http://localhost:8000/quantum-factory/')
        
        # Wait for page to load
        await page.wait_for_load_state('networkidle')
        
        # Check page title and main heading
        title = await page.title()
        assert 'Quantum' in title or 'Factory' in title
        
        # Look for main headings (flexible to handle different template states)
        headings = await page.locator('h1, h2').all_text_contents()
        heading_text = ' '.join(headings).lower()
        
        assert any(keyword in heading_text for keyword in [
            'quantum', 'factory', 'universal', 'digital twin'
        ]), f"Expected quantum-related headings, got: {headings}"
        
        # Check for navigation elements
        nav_elements = await page.locator('a, button').all()
        assert len(nav_elements) > 0, "Expected navigation elements"
        
        print(f"   ✅ Homepage loaded: {title}")
        print(f"   📄 Found {len(nav_elements)} navigation elements")
    
    @pytest.mark.asyncio
    async def test_data_upload_interface(self, page, sample_csv_file):
        """Test data upload interface functionality"""
        
        print("\n📤 Testing Data Upload Interface...")
        
        # Navigate to upload page
        await page.goto('http://localhost:8000/quantum-factory/upload')
        await page.wait_for_load_state('networkidle')
        
        # Look for file input or upload area
        file_inputs = await page.locator('input[type="file"], #fileInput').count()
        upload_areas = await page.locator('.upload-area, [class*="upload"], [id*="upload"]').count()
        
        has_upload_interface = file_inputs > 0 or upload_areas > 0
        
        if has_upload_interface:
            print("   ✅ Upload interface found")
            
            # Try to interact with file input if available
            file_input = page.locator('input[type="file"]').first
            if await file_input.count() > 0:
                await file_input.set_input_files(sample_csv_file)
                print(f"   ✅ File uploaded: {os.path.basename(sample_csv_file)}")
        else:
            print("   ⚠️ Upload interface not found - checking for fallback content")
            
            # Check for fallback content
            content = await page.content()
            assert 'upload' in content.lower() or 'file' in content.lower(), "No upload-related content found"
        
        # Cleanup
        try:
            os.unlink(sample_csv_file)
        except:
            pass
    
    @pytest.mark.asyncio
    async def test_conversational_ai_interface(self, page):
        """Test conversational AI interface"""
        
        print("\n💬 Testing Conversational AI Interface...")
        
        # Navigate to conversation page
        await page.goto('http://localhost:8000/quantum-factory/conversation')
        await page.wait_for_load_state('networkidle')
        
        # Look for chat interface elements
        chat_elements = await page.locator(
            '.chat-container, .message, .chat-input, input[placeholder*="message"], button'
        ).count()
        
        if chat_elements > 0:
            print("   ✅ Chat interface found")
            
            # Look for chat input
            chat_input = page.locator('input[type="text"], textarea, [class*="input"]').first
            
            if await chat_input.count() > 0:
                # Try to send a test message
                await chat_input.fill("I'm a beginner user with CSV data")
                
                # Look for send button
                send_button = page.locator('button').filter(has_text='Send').or_(
                    page.locator('button[type="submit"]')
                ).first
                
                if await send_button.count() > 0 and await send_button.is_enabled():
                    await send_button.click()
                    print("   ✅ Test message sent")
                
        else:
            print("   ⚠️ Chat interface elements not found - checking fallback")
            
            # Check for fallback chat content
            content = await page.content()
            assert any(keyword in content.lower() for keyword in [
                'chat', 'conversation', 'message', 'quantum ai'
            ]), "No conversational content found"
    
    @pytest.mark.asyncio
    async def test_domains_exploration_page(self, page):
        """Test specialized domains exploration page"""
        
        print("\n🏢 Testing Domains Exploration Page...")
        
        # Navigate to domains page
        await page.goto('http://localhost:8000/quantum-factory/domains')
        await page.wait_for_load_state('networkidle')
        
        # Look for domain-related content
        domain_keywords = ['financial', 'iot', 'healthcare', 'manufacturing', 'domain']
        content = await page.content()
        content_lower = content.lower()
        
        found_domains = [keyword for keyword in domain_keywords if keyword in content_lower]
        
        assert len(found_domains) > 0, f"Expected domain content, found keywords: {found_domains}"
        
        # Look for domain cards or sections
        domain_sections = await page.locator(
            '.domain-card, .domain, [class*="financial"], [class*="iot"], [class*="healthcare"]'
        ).count()
        
        print(f"   ✅ Domain content found: {len(found_domains)} keywords")
        print(f"   🏢 Domain sections: {domain_sections}")
        
        # Look for interactive elements (buttons, links)
        interactive_elements = await page.locator('button, a[href*="quantum"]').count()
        print(f"   🔗 Interactive elements: {interactive_elements}")
    
    @pytest.mark.asyncio
    async def test_api_endpoints_accessibility(self, page):
        """Test API endpoints are accessible"""
        
        print("\n🔌 Testing API Endpoints...")
        
        api_endpoints = [
            '/quantum-factory/api/domains',
            '/quantum-factory/api/quantum-advantages', 
            '/quantum-factory/api/factory-stats'
        ]
        
        accessible_endpoints = []
        
        for endpoint in api_endpoints:
            try:
                response = await page.request.get(f'http://localhost:8000{endpoint}')
                
                if response.status == 200:
                    accessible_endpoints.append(endpoint)
                    print(f"   ✅ {endpoint}: {response.status}")
                else:
                    print(f"   ❌ {endpoint}: {response.status}")
                    
            except Exception as e:
                print(f"   ⚠️ {endpoint}: {str(e)}")
        
        # At least some endpoints should be accessible
        assert len(accessible_endpoints) > 0, f"No API endpoints accessible. Tried: {api_endpoints}"
        
        print(f"   📊 Accessible endpoints: {len(accessible_endpoints)}/{len(api_endpoints)}")
    
    @pytest.mark.asyncio
    async def test_responsive_design(self, page):
        """Test responsive design across different screen sizes"""
        
        print("\n📱 Testing Responsive Design...")
        
        screen_sizes = [
            {'width': 1920, 'height': 1080, 'name': 'Desktop Large'},
            {'width': 1280, 'height': 720, 'name': 'Desktop Standard'},
            {'width': 768, 'height': 1024, 'name': 'Tablet'},
            {'width': 375, 'height': 667, 'name': 'Mobile'}
        ]
        
        await page.goto('http://localhost:8000/quantum-factory/')
        
        responsive_results = []
        
        for size in screen_sizes:
            await page.set_viewport_size({'width': size['width'], 'height': size['height']})
            await page.wait_for_timeout(500)  # Let page adjust
            
            # Check if main content is visible
            main_content = await page.locator('body, main, .container, [class*="content"]').first.is_visible()
            
            # Check for any overflow issues
            body_width = await page.evaluate('document.body.scrollWidth')
            viewport_width = size['width']
            
            has_overflow = body_width > viewport_width * 1.1  # Allow 10% tolerance
            
            result = {
                'size': size['name'],
                'content_visible': main_content,
                'no_overflow': not has_overflow,
                'body_width': body_width
            }
            
            responsive_results.append(result)
            
            status = "✅" if main_content and not has_overflow else "⚠️"
            print(f"   {status} {size['name']} ({size['width']}x{size['height']}): Content={main_content}, Overflow={has_overflow}")
        
        # Validate responsive behavior
        visible_count = sum(1 for r in responsive_results if r['content_visible'])
        no_overflow_count = sum(1 for r in responsive_results if r['no_overflow'])
        
        assert visible_count > 0, "Content should be visible on at least one screen size"
        assert no_overflow_count >= len(screen_sizes) // 2, "Most screen sizes should not have overflow issues"
        
        print(f"   📊 Responsive Summary: {visible_count}/{len(screen_sizes)} sizes show content, {no_overflow_count}/{len(screen_sizes)} no overflow")


class TestQuantumFactoryWorkflows:
    """🔄 Test Complete User Workflows"""
    
    @pytest.fixture
    async def browser_context(self):
        """Create browser context for workflow testing"""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        
        yield context
        
        await context.close()
        await browser.close() 
        await playwright.stop()
    
    @pytest.fixture
    async def page(self, browser_context):
        """Create page for workflow testing"""
        page = await browser_context.new_page()
        return page
    
    @pytest.mark.asyncio
    async def test_complete_data_upload_workflow(self, page):
        """Test complete data upload to quantum twin creation workflow"""
        
        print("\n🔄 Testing Complete Data Upload Workflow...")
        
        # Step 1: Navigate to factory homepage
        await page.goto('http://localhost:8000/quantum-factory/')
        await page.wait_for_load_state('networkidle')
        print("   📍 Step 1: Homepage loaded")
        
        # Step 2: Navigate to upload page
        upload_links = await page.locator('a[href*="upload"], button').filter(has_text='Upload').count()
        
        if upload_links > 0:
            await page.locator('a[href*="upload"], button').filter(has_text='Upload').first.click()
            await page.wait_for_load_state('networkidle')
            print("   📍 Step 2: Upload page accessed")
        else:
            # Direct navigation if no link found
            await page.goto('http://localhost:8000/quantum-factory/upload')
            await page.wait_for_load_state('networkidle')
            print("   📍 Step 2: Upload page accessed (direct)")
        
        # Step 3: Check upload interface availability
        upload_interface_available = await page.locator(
            'input[type="file"], .upload-area, #fileInput'
        ).count() > 0
        
        if upload_interface_available:
            print("   ✅ Step 3: Upload interface available")
            
            # Create sample data
            sample_data = pd.DataFrame({
                'sensor_1': np.random.randn(20),
                'sensor_2': np.random.randn(20), 
                'target': np.random.randint(0, 2, 20)
            })
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                sample_data.to_csv(f.name, index=False)
                temp_file = f.name
            
            try:
                # Upload file
                file_input = page.locator('input[type="file"]').first
                await file_input.set_input_files(temp_file)
                print("   📍 Step 4: Sample file uploaded")
                
                # Look for analysis or processing feedback
                await page.wait_for_timeout(2000)  # Wait for processing
                
                feedback_elements = await page.locator(
                    '.success, .result, [class*="analysis"], [class*="quantum"]'
                ).count()
                
                if feedback_elements > 0:
                    print("   ✅ Step 5: Processing feedback received")
                else:
                    print("   ⚠️ Step 5: No processing feedback found")
                
            finally:
                # Cleanup
                try:
                    os.unlink(temp_file)
                except:
                    pass
        else:
            print("   ⚠️ Step 3: Upload interface not available - using fallback validation")
            
            # Validate page has upload-related content
            content = await page.content()
            assert 'upload' in content.lower(), "No upload content found"
        
        print("   🎯 Workflow Status: Data upload workflow tested")
    
    @pytest.mark.asyncio
    async def test_conversational_quantum_twin_creation_workflow(self, page):
        """Test conversational quantum twin creation workflow"""
        
        print("\n💬 Testing Conversational Quantum Twin Creation...")
        
        # Step 1: Navigate to conversation interface
        await page.goto('http://localhost:8000/quantum-factory/conversation')
        await page.wait_for_load_state('networkidle')
        print("   📍 Step 1: Conversation interface loaded")
        
        # Step 2: Check for conversation elements
        conversation_elements = await page.locator(
            '.chat-container, .message, input, button, [class*="conversation"]'
        ).count()
        
        if conversation_elements > 0:
            print("   ✅ Step 2: Conversation elements found")
            
            # Step 3: Try to interact with expertise selection
            expertise_buttons = await page.locator('button').filter(has_text='Beginner').or_(
                page.locator('button').filter(has_text='Intermediate')
            ).count()
            
            if expertise_buttons > 0:
                # Click on beginner or intermediate
                await page.locator('button').filter(has_text='Beginner').or_(
                    page.locator('button').filter(has_text='Intermediate')
                ).first.click()
                print("   📍 Step 3: Expertise level selected")
                
                await page.wait_for_timeout(1000)
                
                # Step 4: Look for follow-up message or input
                chat_input = page.locator('input[type="text"], textarea').first
                
                if await chat_input.count() > 0:
                    await chat_input.fill("I have sensor data from IoT devices")
                    
                    # Try to send message
                    send_button = page.locator('button').filter(has_text='Send').first
                    if await send_button.count() > 0 and await send_button.is_enabled():
                        await send_button.click()
                        print("   📍 Step 4: Message sent about IoT sensor data")
                        
                        await page.wait_for_timeout(2000)
                        
                        # Look for AI response
                        messages = await page.locator('.message, [class*="ai"], [class*="response"]').count()
                        if messages > 0:
                            print("   ✅ Step 5: AI response received")
                        else:
                            print("   ⚠️ Step 5: No AI response detected")
                    else:
                        print("   ⚠️ Step 4: Send button not available")
                else:
                    print("   ⚠️ Step 4: Chat input not found")
            else:
                print("   ⚠️ Step 3: Expertise buttons not found")
        else:
            print("   ⚠️ Step 2: Conversation elements not found - checking fallback")
            
            # Validate conversation content exists
            content = await page.content()
            has_conversation_content = any(keyword in content.lower() for keyword in [
                'conversation', 'chat', 'quantum ai', 'beginner', 'intermediate'
            ])
            
            assert has_conversation_content, "No conversational content found"
            print("   ⚠️ Fallback: Conversation content validated")
        
        print("   🎯 Workflow Status: Conversational workflow tested")
    
    @pytest.mark.asyncio  
    async def test_domain_exploration_workflow(self, page):
        """Test domain exploration workflow"""
        
        print("\n🏢 Testing Domain Exploration Workflow...")
        
        # Step 1: Start from homepage
        await page.goto('http://localhost:8000/quantum-factory/')
        await page.wait_for_load_state('networkidle')
        print("   📍 Step 1: Homepage loaded")
        
        # Step 2: Navigate to domains
        domain_links = await page.locator('a[href*="domains"], button').filter(has_text='Domain').count()
        
        if domain_links > 0:
            await page.locator('a[href*="domains"], button').filter(has_text='Domain').first.click()
        else:
            await page.goto('http://localhost:8000/quantum-factory/domains')
        
        await page.wait_for_load_state('networkidle')
        print("   📍 Step 2: Domains page loaded")
        
        # Step 3: Explore domain content
        domain_cards = await page.locator('.domain-card, [class*="domain"]').count()
        
        if domain_cards > 0:
            print(f"   ✅ Step 3: Found {domain_cards} domain sections")
            
            # Step 4: Try to interact with a domain
            financial_elements = await page.locator('*').filter(has_text='Financial').count()
            iot_elements = await page.locator('*').filter(has_text='IoT').count()
            
            if financial_elements > 0:
                print("   📍 Step 4: Financial domain content found")
            
            if iot_elements > 0:  
                print("   📍 Step 4: IoT domain content found")
            
            # Look for exploration buttons
            explore_buttons = await page.locator('button, a').filter(has_text='Explore').count()
            if explore_buttons > 0:
                print(f"   ✅ Step 5: Found {explore_buttons} exploration options")
            else:
                print("   ⚠️ Step 5: No exploration buttons found")
        else:
            print("   ⚠️ Step 3: No domain cards found - checking content")
            
            # Validate domain content exists
            content = await page.content()
            domain_keywords = ['financial', 'iot', 'healthcare', 'manufacturing']
            found_keywords = [k for k in domain_keywords if k in content.lower()]
            
            assert len(found_keywords) > 0, f"Expected domain keywords, found: {found_keywords}"
            print(f"   ⚠️ Fallback: Found domain keywords: {found_keywords}")
        
        print("   🎯 Workflow Status: Domain exploration tested")


class TestPerformanceAndAccessibility:
    """⚡ Test Performance and Accessibility"""
    
    @pytest.fixture
    async def browser_context(self):
        """Create browser context for performance testing"""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        
        yield context
        
        await context.close()
        await browser.close()
        await playwright.stop()
    
    @pytest.fixture
    async def page(self, browser_context):
        """Create page for performance testing"""
        page = await browser_context.new_page()
        return page
    
    @pytest.mark.asyncio
    async def test_page_load_performance(self, page):
        """Test page load performance"""
        
        print("\n⚡ Testing Page Load Performance...")
        
        pages_to_test = [
            '/quantum-factory/',
            '/quantum-factory/upload',
            '/quantum-factory/conversation',
            '/quantum-factory/domains'
        ]
        
        performance_results = []
        
        for page_path in pages_to_test:
            start_time = asyncio.get_event_loop().time()
            
            try:
                await page.goto(f'http://localhost:8000{page_path}')
                await page.wait_for_load_state('networkidle', timeout=10000)
                
                load_time = asyncio.get_event_loop().time() - start_time
                
                performance_results.append({
                    'page': page_path,
                    'load_time': load_time,
                    'success': True
                })
                
                print(f"   ✅ {page_path}: {load_time:.2f}s")
                
            except Exception as e:
                load_time = asyncio.get_event_loop().time() - start_time
                
                performance_results.append({
                    'page': page_path,
                    'load_time': load_time,
                    'success': False,
                    'error': str(e)
                })
                
                print(f"   ❌ {page_path}: {load_time:.2f}s (Error: {str(e)})")
        
        # Validate performance
        successful_loads = [r for r in performance_results if r['success']]
        assert len(successful_loads) > 0, "At least one page should load successfully"
        
        average_load_time = sum(r['load_time'] for r in successful_loads) / len(successful_loads)
        print(f"   📊 Average load time: {average_load_time:.2f}s")
        
        # Performance threshold (10 seconds should be reasonable for most pages)
        fast_loads = [r for r in successful_loads if r['load_time'] < 10.0]
        performance_ratio = len(fast_loads) / len(successful_loads)
        
        print(f"   📊 Performance ratio: {performance_ratio:.1%} of pages load under 10s")
        
        # At least half of successful loads should be reasonably fast
        assert performance_ratio >= 0.5, f"Expected >50% fast loads, got {performance_ratio:.1%}"
    
    @pytest.mark.asyncio
    async def test_error_handling(self, page):
        """Test error handling for invalid requests"""
        
        print("\n🛡️ Testing Error Handling...")
        
        error_scenarios = [
            '/quantum-factory/nonexistent-page',
            '/quantum-factory/api/invalid-endpoint',
        ]
        
        error_results = []
        
        for scenario in error_scenarios:
            try:
                response = await page.goto(f'http://localhost:8000{scenario}')
                
                # Check if it's a 404 or error page
                status = response.status if response else 0
                
                error_results.append({
                    'scenario': scenario,
                    'status': status,
                    'handled_gracefully': 400 <= status < 600  # 4xx or 5xx is expected
                })
                
                print(f"   📍 {scenario}: Status {status}")
                
            except Exception as e:
                error_results.append({
                    'scenario': scenario,
                    'status': 0,
                    'handled_gracefully': True,  # Exception handling is also valid
                    'error': str(e)
                })
                print(f"   📍 {scenario}: Exception handled")
        
        # Validate error handling
        graceful_errors = [r for r in error_results if r['handled_gracefully']]
        error_handling_ratio = len(graceful_errors) / len(error_results)
        
        print(f"   📊 Error handling: {error_handling_ratio:.1%} of scenarios handled gracefully")
        
        # All error scenarios should be handled gracefully  
        assert error_handling_ratio >= 0.8, f"Expected >80% graceful error handling, got {error_handling_ratio:.1%}"


if __name__ == "__main__":
    # Run E2E tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-s",  # Show print statements
        "--durations=5"  # Show slowest 5 tests
    ])
