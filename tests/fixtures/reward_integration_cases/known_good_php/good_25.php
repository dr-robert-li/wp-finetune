<?php
/**
 * AJAX Handler 25
 *
 * @package WPFinetune\\Tests\\Fixtures\\KnownGood
 */

namespace WPFinetune\\Tests\\Fixtures\\KnownGood;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Handles authenticated AJAX requests.
 */
class Ajax_Handler_25 {{

	/**
	 * @var string Nonce action.
	 */
	const NONCE_ACTION = 'wpf_ajax_25';

	/**
	 * Register AJAX hooks.
	 *
	 * @return void
	 */
	public function register(): void {{
		add_action( 'wp_ajax_wpf_action_25', array( $this, 'handle_request' ) );
	}}

	/**
	 * Handle AJAX request.
	 *
	 * @return void
	 */
	public function handle_request(): void {{
		check_ajax_referer( self::NONCE_ACTION, 'nonce' );

		if ( ! current_user_can( 'edit_posts' ) ) {{
			wp_send_json_error( array( 'message' => __( 'Insufficient permissions.', 'wpf-tests' ) ), 403 );
		}}

		$raw_value = isset( $_POST['value'] ) ? sanitize_text_field( wp_unslash( $_POST['value'] ) ) : '';

		if ( empty( $raw_value ) ) {{
			wp_send_json_error( array( 'message' => __( 'Value is required.', 'wpf-tests' ) ), 400 );
		}}

		wp_send_json_success(
			array(
				'processed' => esc_html( $raw_value ),
				'idx'       => 25,
			)
		);
	}}
}}
