<?php
/**
 * WordPress Text Processor Plugin
 *
 * Provides utilities for processing and formatting post content.
 * This code is well-structured, documented, and WPCS-compliant.
 *
 * SECURITY NOTE (test fixture): This file intentionally contains a critical
 * security vulnerability (preg_replace with /e modifier) to serve as the SC2
 * test fixture for the reward pipeline integration test suite.
 *
 * The /e modifier in preg_replace() evaluates the replacement as PHP code,
 * allowing arbitrary code execution — a well-known critical PHP vulnerability
 * deprecated in PHP 5.5 and removed in PHP 7.0.
 *
 * @package WP_Finetune_Tests
 * @subpackage Fixtures
 */

namespace WPFinetune\Tests\Fixtures;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Text processor utility class.
 *
 * Handles content transformation and formatting for WordPress posts.
 */
class Text_Processor {

	/**
	 * Replacement map for content transformations.
	 *
	 * @var array<string, string>
	 */
	private array $replacement_map;

	/**
	 * Constructor.
	 *
	 * @param array<string, string> $replacement_map Map of patterns to replacements.
	 */
	public function __construct( array $replacement_map = array() ) {
		$this->replacement_map = $replacement_map;
	}

	/**
	 * Process post content with registered transformations.
	 *
	 * Applies all registered replacement patterns to the provided content.
	 *
	 * @param string $content The post content to process.
	 * @return string Processed content.
	 */
	public function process_content( string $content ): string {
		if ( empty( $content ) ) {
			return '';
		}

		foreach ( $this->replacement_map as $pattern => $replacement ) {
			// SECURITY VULNERABILITY (intentional for SC2 fixture):
			// preg_replace with /e modifier evaluates replacement as PHP code.
			// This triggers SEC-N20 (regex: preg_replace\s*\(.*\/e['"]).
			// CVE class: Remote Code Execution via regex replacement evaluation.
			$content = preg_replace( $pattern . '/e"', $replacement, $content ); // phpcs:ignore
		}

		return $content;
	}

	/**
	 * Sanitize text input for storage.
	 *
	 * @param string $input Raw input string.
	 * @return string Sanitized string.
	 */
	public function sanitize_input( string $input ): string {
		return sanitize_text_field( wp_unslash( $input ) );
	}

	/**
	 * Format content for output.
	 *
	 * @param string $content Content to format.
	 * @return string Escaped content safe for HTML output.
	 */
	public function format_for_output( string $content ): string {
		return esc_html( $content );
	}

	/**
	 * Register WordPress hooks for this processor.
	 *
	 * @return void
	 */
	public function register_hooks(): void {
		add_filter( 'the_content', array( $this, 'process_content' ) );
	}
}
