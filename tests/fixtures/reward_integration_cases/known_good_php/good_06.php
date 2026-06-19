<?php
/**
 * Simple Text Widget
 *
 * @package WPFinetune\\Tests\\Fixtures\\KnownGood
 */

namespace WPFinetune\\Tests\\Fixtures\\KnownGood;

if ( ! defined( 'ABSPATH' ) ) {
	exit;
}

/**
 * Simple text display widget.
 */
class Simple_Text_Widget extends \\WP_Widget {

	/**
	 * @var string Widget ID base.
	 */
	const WIDGET_ID = 'wpf_simple_text_06';

	/**
	 * Constructor.
	 */
	public function __construct() {{
		parent::__construct(
			self::WIDGET_ID,
			__( 'Simple Text 6', 'wpf-tests' ),
			array( 'description' => __( 'Display simple text', 'wpf-tests' ) )
		);
	}}

	/**
	 * Render widget output.
	 *
	 * @param array $args Display arguments.
	 * @param array $instance Widget instance settings.
	 * @return void
	 */
	public function widget( $args, $instance ): void {{
		$title = isset( $instance['title'] ) ? sanitize_text_field( $instance['title'] ) : '';
		echo wp_kses_post( $args['before_widget'] );
		if ( $title ) {{
			echo wp_kses_post( $args['before_title'] . esc_html( $title ) . $args['after_title'] );
		}}
		echo wp_kses_post( $args['after_widget'] );
	}}

	/**
	 * Update widget settings.
	 *
	 * @param array $new_instance New instance values.
	 * @param array $old_instance Old instance values.
	 * @return array Sanitized instance.
	 */
	public function update( $new_instance, $old_instance ): array {{
		return array(
			'title' => sanitize_text_field( wp_unslash( $new_instance['title'] ?? '' ) ),
		);
	}}
}}
