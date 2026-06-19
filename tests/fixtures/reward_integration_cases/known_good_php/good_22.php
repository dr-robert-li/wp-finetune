<?php
/**
 * REST API Endpoint 22
 *
 * @package WPFinetune\\Tests\\Fixtures\\KnownGood
 */

namespace WPFinetune\\Tests\\Fixtures\\KnownGood;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Items REST controller.
 */
class Items_Controller_22 extends \\WP_REST_Controller {{

	/**
	 * @var string REST namespace.
	 */
	protected string $namespace = 'wpf/v1';

	/**
	 * @var string REST base.
	 */
	protected string $rest_base = 'items-22';

	/**
	 * Register routes.
	 *
	 * @return void
	 */
	public function register_routes(): void {{
		register_rest_route(
			$this->namespace,
			'/items-22',
			array(
				array(
					'methods'             => \\WP_REST_Server::READABLE,
					'callback'            => array( $this, 'get_items' ),
					'permission_callback' => array( $this, 'get_items_permissions_check' ),
				),
			)
		);
	}}

	/**
	 * Permission check for GET items.
	 *
	 * @param \\WP_REST_Request $request Request object.
	 * @return bool
	 */
	public function get_items_permissions_check( $request ): bool {{
		return current_user_can( 'read' );
	}}

	/**
	 * Get items collection.
	 *
	 * @param \\WP_REST_Request $request Request object.
	 * @return \\WP_REST_Response
	 */
	public function get_items( $request ): \\WP_REST_Response {{
		$per_page = absint( $request->get_param( 'per_page' ) ?? 10 );
		$page     = absint( $request->get_param( 'page' ) ?? 1 );
		return new \\WP_REST_Response(
			array(
				'items'   => array(),
				'page'    => $page,
				'per_page' => $per_page,
			),
			200
		);
	}}
}}
