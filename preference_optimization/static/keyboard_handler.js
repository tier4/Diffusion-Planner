/**
 * Keyboard Handler for Trajectory Preference Annotation GUI
 * 
 * This script sets up global keyboard event listeners for arrow key navigation.
 * It captures left and right arrow key presses and dispatches them to a hidden
 * Gradio textbox component for processing by the Python backend.
 * 
 * Each keypress is assigned a unique counter to ensure Gradio's change event
 * triggers consistently, even for consecutive presses of the same key.
 */

(function() {
    console.log('Setting up keyboard listener...');
    
    // Only add listener once
    if (!window.keyboardListenerAdded) {
        // Counter to ensure unique values for each keypress
        let keypressCounter = 0;
        
        document.addEventListener('keydown', (e) => {
            console.log('Key pressed:', e.key);
            
            // Only handle left and right arrow keys
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
                e.preventDefault();
                console.log('Arrow key detected:', e.key);
                
                // Find the keyboard capture component container
                const container = document.getElementById('keyboard_capture');
                console.log('Keyboard capture container:', container);
                
                if (container) {
                    // Find the actual input element inside the Gradio component
                    const input = container.querySelector('textarea') || 
                                  container.querySelector('input[type="text"]');
                    console.log('Found input element:', input);
                    
                    if (input) {
                        // Set value with counter to ensure uniqueness
                        // Format: "ArrowLeft:123" or "ArrowRight:456"
                        keypressCounter++;
                        const value = e.key + ':' + keypressCounter;
                        input.value = value;
                        console.log('Set value to:', value);
                        
                        // Trigger input event to notify Gradio
                        const inputEvent = new Event('input', {bubbles: true});
                        input.dispatchEvent(inputEvent);
                        console.log('Dispatched input event');
                        
                        // Also trigger change event for compatibility
                        const changeEvent = new Event('change', {bubbles: true});
                        input.dispatchEvent(changeEvent);
                        console.log('Dispatched change event for:', value);
                    } else {
                        console.error('Could not find input element inside keyboard_capture');
                    }
                } else {
                    console.error('Could not find keyboard_capture container');
                }
            }
        });
        
        window.keyboardListenerAdded = true;
        console.log('Keyboard listener added successfully');
    }
})();

